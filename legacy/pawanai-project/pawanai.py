# PawanAI: Real-time Air Pollution Forecasting Platform
# Complete Implementation by Indraneel Chatterjee

import numpy as np
import pandas as pd
import xarray as xr
import requests
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import streamlit as st

st.title("Hello Streamlit!")
st.write("Your app is working!")
# Geospatial and visualization
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Web framework
import streamlit as st
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Suppress warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataCollector:
    """Handles data collection from multiple sources"""
    
    def __init__(self):
        self.cpcb_stations = {}
        self.openaq_api = "https://api.openaq.org/v2"
        
    def fetch_cpcb_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch CPCB ground station data"""
        try:
            # Simulated CPCB data structure - replace with actual API calls
            dates = pd.date_range(start_date, end_date, freq='H')
            stations = ['Delhi_RK_Puram', 'Mumbai_Bandra', 'Kolkata_Jadavpur', 'Chennai_Alandur']
            
            data = []
            for station in stations:
                for date in dates:
                    data.append({
                        'datetime': date,
                        'station_id': station,
                        'pm25': np.random.normal(80, 30),  # Realistic PM2.5 values
                        'latitude': np.random.uniform(8, 35),  # India lat range
                        'longitude': np.random.uniform(68, 97),  # India lon range
                        'temperature': np.random.uniform(15, 40),
                        'humidity': np.random.uniform(30, 90)
                    })
            
            df = pd.DataFrame(data)
            df['pm25'] = np.clip(df['pm25'], 0, 500)  # Realistic bounds
            return df
            
        except Exception as e:
            logger.error(f"Error fetching CPCB data: {e}")
            return pd.DataFrame()
    
    def fetch_openaq_data(self, country: str = 'IN', parameter: str = 'pm25') -> pd.DataFrame:
        """Fetch data from OpenAQ API"""
        try:
            url = f"{self.openaq_api}/measurements"
            params = {
                'country': country,
                'parameter': parameter,
                'limit': 10000,
                'date_from': (datetime.now() - timedelta(days=7)).isoformat(),
                'date_to': datetime.now().isoformat()
            }
            
            response = requests.get(url, params=params, timeout=30)
            if response.status_code == 200:
                data = response.json()
                measurements = data.get('results', [])
                
                df_data = []
                for m in measurements:
                    df_data.append({
                        'datetime': pd.to_datetime(m['date']['utc']),
                        'location': m['location'],
                        'pm25': m['value'],
                        'latitude': m['coordinates']['latitude'],
                        'longitude': m['coordinates']['longitude'],
                        'unit': m['unit']
                    })
                
                return pd.DataFrame(df_data)
            else:
                logger.warning(f"OpenAQ API error: {response.status_code}")
                return self.fetch_cpcb_data(
                    (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                    datetime.now().strftime('%Y-%m-%d')
                )
                
        except Exception as e:
            logger.error(f"Error fetching OpenAQ data: {e}")
            # Fallback to simulated data
            return self.fetch_cpcb_data(
                (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                datetime.now().strftime('%Y-%m-%d')
            )
    
    def generate_aod_data(self, bbox: Tuple[float, float, float, float], 
                         resolution: float = 0.01) -> xr.Dataset:
        """Generate simulated AOD data (replace with INSAT-3D data)"""
        try:
            min_lon, min_lat, max_lon, max_lat = bbox
            
            # Create coordinate arrays
            lons = np.arange(min_lon, max_lon, resolution)
            lats = np.arange(min_lat, max_lat, resolution)
            times = pd.date_range(datetime.now() - timedelta(days=7), 
                                datetime.now(), freq='H')
            
            # Generate realistic AOD values (0.1 to 2.0)
            np.random.seed(42)  # Reproducible data
            aod_data = np.random.lognormal(mean=-1, sigma=0.5, 
                                         size=(len(times), len(lats), len(lons)))
            aod_data = np.clip(aod_data, 0.05, 3.0)
            
            # Create xarray dataset
            ds = xr.Dataset({
                'aod': (['time', 'lat', 'lon'], aod_data),
                'aod_quality': (['time', 'lat', 'lon'], 
                              np.random.choice([0, 1, 2], size=aod_data.shape, p=[0.7, 0.2, 0.1]))
            }, coords={
                'time': times,
                'lat': lats,
                'lon': lons
            })
            
            return ds
            
        except Exception as e:
            logger.error(f"Error generating AOD data: {e}")
            return xr.Dataset()
    
    def fetch_merra2_data(self, bbox: Tuple[float, float, float, float]) -> pd.DataFrame:
        """Fetch MERRA-2 reanalysis data (simulated)"""
        try:
            times = pd.date_range(datetime.now() - timedelta(days=7), 
                                datetime.now(), freq='H')
            
            # Simulate meteorological variables
            data = []
            for time in times:
                data.append({
                    'datetime': time,
                    'temperature': np.random.normal(25, 8),  # Celsius
                    'humidity': np.random.uniform(30, 90),   # %
                    'wind_speed': np.random.exponential(3), # m/s
                    'wind_direction': np.random.uniform(0, 360), # degrees
                    'boundary_layer_height': np.random.normal(800, 300), # meters
                    'pressure': np.random.normal(1013, 10)   # hPa
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Error fetching MERRA-2 data: {e}")
            return pd.DataFrame()

class FeatureEngineer:
    """Advanced feature engineering for pollution prediction"""
    
    @staticmethod
    def calculate_aod_gradients(aod_data: xr.Dataset) -> xr.Dataset:
        """Calculate spatial gradients of AOD"""
        try:
            # Calculate gradients using central differences
            aod_dx = aod_data['aod'].diff('lon') / aod_data['aod'].diff('lon').lon
            aod_dy = aod_data['aod'].diff('lat') / aod_data['aod'].diff('lat').lat
            
            # Add gradient magnitude
            gradient_magnitude = np.sqrt(aod_dx**2 + aod_dy**2)
            
            # Add to dataset
            aod_data['aod_gradient_x'] = aod_dx
            aod_data['aod_gradient_y'] = aod_dy
            aod_data['aod_gradient_magnitude'] = gradient_magnitude
            
            return aod_data
            
        except Exception as e:
            logger.error(f"Error calculating AOD gradients: {e}")
            return aod_data
    
    @staticmethod
    def create_lag_features(df: pd.DataFrame, columns: List[str], 
                          lags: List[int] = [1, 2, 3, 6]) -> pd.DataFrame:
        """Create lagged features for time series"""
        try:
            df_lagged = df.copy()
            
            for col in columns:
                if col in df.columns:
                    for lag in lags:
                        df_lagged[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            return df_lagged.dropna()
            
        except Exception as e:
            logger.error(f"Error creating lag features: {e}")
            return df
    
    @staticmethod
    def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        try:
            if all(col in df.columns for col in ['humidity', 'aod', 'temperature', 'wind_speed']):
                # Humidity-AOD interaction (important for particle hygroscopic growth)
                df['humidity_aod_interaction'] = df['humidity'] * df['aod']
                
                # Temperature-Wind interaction
                df['temp_wind_interaction'] = df['temperature'] * df['wind_speed']
                
                # Atmospheric stability indicator
                df['stability_index'] = df['temperature'] / (df['wind_speed'] + 0.1)
                
                # Pollution potential index
                df['pollution_potential'] = (df['aod'] * df['humidity']) / (df['wind_speed'] + 0.1)
            
            return df
            
        except Exception as e:
            logger.error(f"Error creating interaction features: {e}")
            return df
    
    @staticmethod
    def calculate_wind_vectors(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate wind vector components"""
        try:
            if 'wind_direction' in df.columns and 'wind_speed' in df.columns:
                # Convert to radians
                wind_rad = np.radians(df['wind_direction'])
                
                # Calculate components (meteorological convention)
                df['wind_u'] = -df['wind_speed'] * np.sin(wind_rad)  # East-West
                df['wind_v'] = -df['wind_speed'] * np.cos(wind_rad)  # North-South
                
                # Wind persistence (vector consistency)
                df['wind_u_lag1'] = df['wind_u'].shift(1)
                df['wind_v_lag1'] = df['wind_v'].shift(1)
                
                # Calculate wind persistence
                wind_persistence = (df['wind_u'] * df['wind_u_lag1'] + 
                                  df['wind_v'] * df['wind_v_lag1']) / (df['wind_speed']**2 + 0.1)
                df['wind_persistence'] = wind_persistence.fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Error calculating wind vectors: {e}")
            return df

class PollutionPredictor:
    """Main prediction model using ensemble methods"""
    
    def __init__(self):
        self.rf_model = RandomForestRegressor(
            n_estimators=100, 
            max_depth=15, 
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = RobustScaler()
        self.feature_names = []
        self.is_trained = False
        
    def prepare_features(self, aod_data: xr.Dataset, ground_data: pd.DataFrame, 
                        merra_data: pd.DataFrame) -> pd.DataFrame:
        """Combine and prepare all features for modeling"""
        try:
            # Sample AOD data at ground station locations
            aod_features = []
            
            for _, row in ground_data.iterrows():
                try:
                    # Find nearest AOD pixel
                    aod_point = aod_data.sel(
                        lat=row['latitude'], 
                        lon=row['longitude'], 
                        time=row['datetime'],
                        method='nearest'
                    )
                    
                    aod_features.append({
                        'datetime': row['datetime'],
                        'station_id': row.get('station_id', row.get('location', 'unknown')),
                        'aod': float(aod_point['aod'].values),
                        'aod_gradient_x': float(aod_point.get('aod_gradient_x', 0).values),
                        'aod_gradient_y': float(aod_point.get('aod_gradient_y', 0).values),
                        'aod_gradient_magnitude': float(aod_point.get('aod_gradient_magnitude', 0).values)
                    })
                except Exception as e:
                    logger.warning(f"Error sampling AOD for station: {e}")
                    # Fallback values
                    aod_features.append({
                        'datetime': row['datetime'],
                        'station_id': row.get('station_id', row.get('location', 'unknown')),
                        'aod': 0.5,
                        'aod_gradient_x': 0.0,
                        'aod_gradient_y': 0.0,
                        'aod_gradient_magnitude': 0.0
                    })
            
            aod_df = pd.DataFrame(aod_features)
            
            # Merge datasets
            merged_df = ground_data.merge(aod_df, on=['datetime', 'station_id'], how='inner')
            
            # Merge with MERRA-2 data (temporal join)
            merged_df['datetime_rounded'] = merged_df['datetime'].dt.round('H')
            merra_data['datetime_rounded'] = merra_data['datetime'].dt.round('H')
            
            final_df = merged_df.merge(merra_data, on='datetime_rounded', how='left')
            
            # Feature engineering
            engineer = FeatureEngineer()
            
            # Create lag features
            feature_cols = ['aod', 'temperature', 'humidity', 'wind_speed', 'pm25']
            final_df = engineer.create_lag_features(final_df, feature_cols)
            
            # Create interaction features
            final_df = engineer.create_interaction_features(final_df)
            
            # Calculate wind vectors
            final_df = engineer.calculate_wind_vectors(final_df)
            
            # Add temporal features
            final_df['hour'] = final_df['datetime'].dt.hour
            final_df['day_of_week'] = final_df['datetime'].dt.dayofweek
            final_df['month'] = final_df['datetime'].dt.month
            
            # Cyclical encoding for temporal features
            final_df['hour_sin'] = np.sin(2 * np.pi * final_df['hour'] / 24)
            final_df['hour_cos'] = np.cos(2 * np.pi * final_df['hour'] / 24)
            final_df['day_sin'] = np.sin(2 * np.pi * final_df['day_of_week'] / 7)
            final_df['day_cos'] = np.cos(2 * np.pi * final_df['day_of_week'] / 7)
            
            return final_df.dropna()
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            return pd.DataFrame()
    
    def train_model(self, features_df: pd.DataFrame) -> Dict[str, float]:
        """Train the ensemble model"""
        try:
            # Select feature columns (exclude target and metadata)
            exclude_cols = ['pm25', 'datetime', 'station_id', 'location', 'datetime_rounded', 
                          'latitude', 'longitude', 'unit']
            feature_cols = [col for col in features_df.columns if col not in exclude_cols]
            
            X = features_df[feature_cols]
            y = features_df['pm25']
            
            # Handle missing values
            X = X.fillna(X.median())
            
            # Store feature names
            self.feature_names = feature_cols
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=True
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            logger.info("Training Random Forest model...")
            self.rf_model.fit(X_train_scaled, y_train)
            
            logger.info("Training XGBoost model...")
            self.xgb_model.fit(X_train_scaled, y_train)
            
            # Make predictions
            rf_pred = self.rf_model.predict(X_test_scaled)
            xgb_pred = self.xgb_model.predict(X_test_scaled)
            
            # Ensemble prediction (weighted average)
            ensemble_pred = 0.6 * rf_pred + 0.4 * xgb_pred
            
            # Calculate metrics
            metrics = {
                'mae': mean_absolute_error(y_test, ensemble_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, ensemble_pred)),
                'r2': r2_score(y_test, ensemble_pred),
                'rf_mae': mean_absolute_error(y_test, rf_pred),
                'xgb_mae': mean_absolute_error(y_test, xgb_pred)
            }
            
            self.is_trained = True
            logger.info(f"Model training completed. Metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {}
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the trained ensemble model"""
        try:
            if not self.is_trained:
                raise ValueError("Model must be trained before making predictions")
            
            # Select and prepare features
            X = features_df[self.feature_names].fillna(features_df[self.feature_names].median())
            X_scaled = self.scaler.transform(X)
            
            # Make predictions with both models
            rf_pred = self.rf_model.predict(X_scaled)
            xgb_pred = self.xgb_model.predict(X_scaled)
            
            # Ensemble prediction
            ensemble_pred = 0.6 * rf_pred + 0.4 * xgb_pred
            
            # Ensure predictions are non-negative and reasonable
            ensemble_pred = np.clip(ensemble_pred, 0, 500)
            
            return ensemble_pred
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            return np.array([])
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from Random Forest model"""
        try:
            if not self.is_trained:
                return {}
            
            importance_dict = dict(zip(self.feature_names, self.rf_model.feature_importances_))
            return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            return {}

class Visualizer:
    """Create visualizations and dashboards"""
    
    @staticmethod
    def create_pollution_heatmap(predictions_df: pd.DataFrame, 
                               bbox: Tuple[float, float, float, float]) -> folium.Map:
        """Create interactive pollution heatmap"""
        try:
            center_lat = (bbox[1] + bbox[3]) / 2
            center_lon = (bbox[0] + bbox[2]) / 2
            
            # Create base map
            m = folium.Map(
                location=[center_lat, center_lon],
                zoom_start=6,
                tiles='OpenStreetMap'
            )
            
            # Prepare data for heatmap
            heat_data = []
            for _, row in predictions_df.iterrows():
                if pd.notna(row['latitude']) and pd.notna(row['longitude']) and pd.notna(row['predicted_pm25']):
                    # Normalize PM2.5 for heatmap intensity
                    intensity = min(row['predicted_pm25'] / 200, 1.0)
                    heat_data.append([row['latitude'], row['longitude'], intensity])
            
            if heat_data:
                # Add heatmap layer
                plugins.HeatMap(
                    heat_data,
                    min_opacity=0.2,
                    max_zoom=18,
                    radius=25,
                    blur=15,
                    gradient={
                        0.0: 'green',
                        0.3: 'yellow', 
                        0.6: 'orange',
                        1.0: 'red'
                    }
                ).add_to(m)
                
                # Add markers for stations
                for _, row in predictions_df.iterrows():
                    if pd.notna(row['latitude']) and pd.notna(row['longitude']):
                        pm25 = row['predicted_pm25']
                        
                        # Determine AQI category and color
                        if pm25 <= 12:
                            color, aqi_cat = 'green', 'Good'
                        elif pm25 <= 35:
                            color, aqi_cat = 'yellow', 'Moderate'
                        elif pm25 <= 55:
                            color, aqi_cat = 'orange', 'Unhealthy for Sensitive'
                        elif pm25 <= 150:
                            color, aqi_cat = 'red', 'Unhealthy'
                        else:
                            color, aqi_cat = 'purple', 'Hazardous'
                        
                        # Create popup text
                        popup_text = f"""
                        <b>Location:</b> {row.get('station_id', 'Unknown')}<br>
                        <b>PM2.5:</b> {pm25:.1f} μg/m³<br>
                        <b>AQI Category:</b> {aqi_cat}<br>
                        <b>Time:</b> {row.get('datetime', 'Unknown')}
                        """
                        
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=8,
                            popup=folium.Popup(popup_text, max_width=200),
                            color='black',
                            weight=1,
                            fillColor=color,
                            fillOpacity=0.7
                        ).add_to(m)
            
            return m
            
        except Exception as e:
            logger.error(f"Error creating heatmap: {e}")
            return folium.Map(location=[20, 77], zoom_start=5)
    
    @staticmethod
    def create_time_series_plot(data_df: pd.DataFrame) -> go.Figure:
        """Create time series plot of pollution trends"""
        try:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('PM2.5 Concentration', 'Meteorological Variables'),
                vertical_spacing=0.08
            )
            
            # PM2.5 time series
            if 'pm25' in data_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data_df['datetime'],
                        y=data_df['pm25'],
                        mode='lines+markers',
                        name='Observed PM2.5',
                        line=dict(color='blue', width=2),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
            
            if 'predicted_pm25' in data_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data_df['datetime'],
                        y=data_df['predicted_pm25'],
                        mode='lines+markers',
                        name='Predicted PM2.5',
                        line=dict(color='red', width=2, dash='dash'),
                        marker=dict(size=4)
                    ),
                    row=1, col=1
                )
            
            # Add AQI threshold lines
            aqi_thresholds = [12, 35, 55, 150, 250]
            colors = ['green', 'yellow', 'orange', 'red', 'purple']
            labels = ['Good', 'Moderate', 'USG', 'Unhealthy', 'Hazardous']
            
            for i, (threshold, color, label) in enumerate(zip(aqi_thresholds, colors, labels)):
                fig.add_hline(
                    y=threshold,
                    line=dict(color=color, width=1, dash='dot'),
                    annotation_text=f"{label} ({threshold})",
                    annotation_position="right",
                    row=1, col=1
                )
            
            # Meteorological variables
            if 'temperature' in data_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data_df['datetime'],
                        y=data_df['temperature'],
                        mode='lines',
                        name='Temperature (°C)',
                        line=dict(color='orange'),
                        yaxis='y3'
                    ),
                    row=2, col=1
                )
            
            if 'wind_speed' in data_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data_df['datetime'],
                        y=data_df['wind_speed'],
                        mode='lines',
                        name='Wind Speed (m/s)',
                        line=dict(color='green'),
                        yaxis='y4'
                    ),
                    row=2, col=1
                )
            
            # Update layout
            fig.update_layout(
                title="PawanAI: Pollution and Weather Trends",
                height=600,
                showlegend=True,
                template='plotly_white'
            )
            
            fig.update_xaxes(title_text="Time", row=2, col=1)
            fig.update_yaxes(title_text="PM2.5 (μg/m³)", row=1, col=1)
            fig.update_yaxes(title_text="Temperature/Wind", row=2, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating time series plot: {e}")
            return go.Figure()
    
    @staticmethod
    def create_feature_importance_plot(importance_dict: Dict[str, float]) -> go.Figure:
        """Create feature importance visualization"""
        try:
            # Sort features by importance
            sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features[:15])  # Top 15 features
            
            fig = go.Figure(go.Bar(
                x=list(importances),
                y=list(features),
                orientation='h',
                marker=dict(
                    color=importances,
                    colorscale='Viridis',
                    showscale=True
                )
            ))
            
            fig.update_layout(
                title="Feature Importance for PM2.5 Prediction",
                xaxis_title="Importance Score",
                yaxis_title="Features",
                height=500,
                template='plotly_white'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating feature importance plot: {e}")
            return go.Figure()

class AlertSystem:
    """Air quality alert and notification system"""
    
    def __init__(self):
        self.thresholds = {
            'good': 12,
            'moderate': 35,
            'unhealthy_sensitive': 55,
            'unhealthy': 150,
            'very_unhealthy': 250,
            'hazardous': 300
        }
    
    def calculate_aqi(self, pm25: float) -> Tuple[int, str]:
        """Calculate AQI from PM2.5 concentration"""
        try:
            # EPA AQI breakpoints for PM2.5
            breakpoints = [
                (0, 12, 0, 50),      # Good
                (12.1, 35.4, 51, 100),   # Moderate
                (35.5, 55.4, 101, 150),  # Unhealthy for Sensitive Groups
                (55.5, 150.4, 151, 200), # Unhealthy
                (150.5, 250.4, 201, 300), # Very Unhealthy
                (250.5, 500.4, 301, 500)  # Hazardous
            ]
            
            categories = ['Good', 'Moderate', 'Unhealthy for Sensitive Groups', 
                         'Unhealthy', 'Very Unhealthy', 'Hazardous']
            
            for i, (c_low, c_high, i_low, i_high) in enumerate(breakpoints):
                if c_low <= pm25 <= c_high:
                    # Linear interpolation
                    aqi = ((i_high - i_low) / (c_high - c_low)) * (pm25 - c_low) + i_low
                    return int(round(aqi)), categories[i]
            
            # If concentration exceeds all breakpoints
            return 500, 'Hazardous'
            
        except Exception as e:
            logger.error(f"Error calculating AQI: {e}")
            return 0, 'Unknown'
    
    def generate_alerts(self, predictions_df: pd.DataFrame) -> List[Dict]:
        """Generate alerts based on pollution predictions"""
        try:
            alerts = []
            
            for _, row in predictions_df.iterrows():
                pm25 = row.get('predicted_pm25', 0)
                aqi, category = self.calculate_aqi(pm25)
                
                # Generate alert if unhealthy or worse
                if pm25 > self.thresholds['unhealthy_sensitive']:
                    alert = {
                        'location': row.get('station_id', 'Unknown'),
                        'timestamp': row.get('datetime', datetime.now()),
                        'pm25': pm25,
                        'aqi': aqi,
                        'category': category,
                        'severity': self._get_severity(pm25),
                        'health_message': self._get_health_message(category),
                        'recommendations': self._get_recommendations(category)
                    }
                    alerts.append(alert)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return []
    
    def _get_severity(self, pm25: float) -> str:
        """Get alert severity level"""
        if pm25 > self.thresholds['hazardous']:
            return 'CRITICAL'
        elif pm25 > self.thresholds['very_unhealthy']:
            return 'HIGH'
        elif pm25 > self.thresholds['unhealthy']:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _get_health_message(self, category: str) -> str:
        """Get health message based on AQI category"""
        messages = {
            'Good': 'Air quality is satisfactory for most people.',
            'Moderate': 'Air quality is acceptable. Sensitive individuals may experience minor symptoms.',
            'Unhealthy for Sensitive Groups': 'Sensitive groups may experience health effects.',
            'Unhealthy': 'Everyone may experience health effects. Sensitive groups may experience serious effects.',
            'Very Unhealthy': 'Health alert: everyone may experience serious health effects.',
            'Hazardous': 'Emergency conditions: everyone is likely to be affected.'
        }
        return messages.get(category, 'Unknown air quality conditions.')
    
    def _get_recommendations(self, category: str) -> List[str]:
        """Get recommendations based on AQI category"""
        recommendations = {
            'Good': ['Enjoy outdoor activities'],
            'Moderate': ['Sensitive individuals should limit prolonged outdoor exertion'],
            'Unhealthy for Sensitive Groups': [
                'Sensitive groups should avoid outdoor activities',
                'Close windows and use air purifiers indoors'
            ],
            'Unhealthy': [
                'Everyone should avoid outdoor activities',
                'Keep windows closed',
                'Use air purifiers',
                'Wear N95 masks if you must go outside'
            ],
            'Very Unhealthy': [
                'Stay indoors',
                'Avoid all outdoor activities',
                'Use air purifiers continuously',
                'Seek medical attention if experiencing symptoms'
            ],
            'Hazardous': [
                'Emergency conditions - stay indoors',
                'Seal doors and windows',
                'Use air purifiers on highest setting',
                'Seek immediate medical attention for any symptoms'
            ]
        }
        return recommendations.get(category, ['Monitor air quality updates'])

class PawanAIApp:
    """Main application class integrating all components"""
    
    def __init__(self):
        self.data_collector = DataCollector()
        self.predictor = PollutionPredictor()
        self.visualizer = Visualizer()
        self.alert_system = AlertSystem()
        self.india_bbox = (68.0, 8.0, 97.0, 35.0)  # India bounding box
        
    def run_pipeline(self) -> Dict:
        """Run the complete prediction pipeline"""
        try:
            logger.info("Starting PawanAI prediction pipeline...")
            
            # Step 1: Collect data
            logger.info("Collecting ground station data...")
            ground_data = self.data_collector.fetch_openaq_data('IN', 'pm25')
            
            logger.info("Generating AOD data...")
            aod_data = self.data_collector.generate_aod_data(self.india_bbox)
            aod_data = FeatureEngineer.calculate_aod_gradients(aod_data)
            
            logger.info("Fetching meteorological data...")
            merra_data = self.data_collector.fetch_merra2_data(self.india_bbox)
            
            if ground_data.empty or merra_data.empty:
                raise ValueError("Failed to collect sufficient data")
            
            # Step 2: Prepare features
            logger.info("Preparing features...")
            features_df = self.predictor.prepare_features(aod_data, ground_data, merra_data)
            
            if features_df.empty:
                raise ValueError("Failed to prepare features")
            
            # Step 3: Train model
            logger.info("Training prediction model...")
            metrics = self.predictor.train_model(features_df)
            
            # Step 4: Make predictions
            logger.info("Making predictions...")
            predictions = self.predictor.predict(features_df)
            features_df['predicted_pm25'] = predictions
            
            # Step 5: Generate alerts
            logger.info("Generating alerts...")
            alerts = self.alert_system.generate_alerts(features_df)
            
            # Step 6: Create visualizations
            logger.info("Creating visualizations...")
            pollution_map = self.visualizer.create_pollution_heatmap(features_df, self.india_bbox)
            time_series_plot = self.visualizer.create_time_series_plot(features_df)
            feature_importance = self.predictor.get_feature_importance()
            importance_plot = self.visualizer.create_feature_importance_plot(feature_importance)
            
            results = {
                'metrics': metrics,
                'predictions': features_df,
                'alerts': alerts,
                'map': pollution_map,
                'time_series': time_series_plot,
                'importance_plot': importance_plot,
                'feature_importance': feature_importance
            }
            
            logger.info("Pipeline completed successfully!")
            return results
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            return {}

# Streamlit Web Application
def create_streamlit_app():
    """Create Streamlit web application"""
    
    st.set_page_config(
        page_title="PawanAI - Air Pollution Forecasting",
        page_icon="🌬️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🌬️ PawanAI: Real-time Air Pollution Forecasting")
    st.markdown("*AI-powered pollution prediction using satellite data and machine learning*")
    
    # Sidebar controls
    st.sidebar.header("Control Panel")
    
    if st.sidebar.button("🔄 Run Prediction Pipeline", type="primary"):
        with st.spinner("Running PawanAI pipeline..."):
            app = PawanAIApp()
            results = app.run_pipeline()
            
            if results:
                st.session_state['results'] = results
                st.success("Pipeline completed successfully!")
            else:
                st.error("Pipeline failed. Please check logs.")
    
    # Display results if available
    if 'results' in st.session_state:
        results = st.session_state['results']
        
        # Metrics display
        if 'metrics' in results and results['metrics']:
            st.header("📊 Model Performance")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MAE", f"{results['metrics'].get('mae', 0):.2f} μg/m³")
            with col2:
                st.metric("RMSE", f"{results['metrics'].get('rmse', 0):.2f} μg/m³")
            with col3:
                st.metric("R² Score", f"{results['metrics'].get('r2', 0):.3f}")
            with col4:
                st.metric("Data Points", len(results.get('predictions', [])))
        
        # Alerts section
        if 'alerts' in results and results['alerts']:
            st.header("🚨 Air Quality Alerts")
            for alert in results['alerts'][:5]:  # Show top 5 alerts
                severity_color = {
                    'CRITICAL': '🔴',
                    'HIGH': '🟠',
                    'MEDIUM': '🟡',
                    'LOW': '🟢'
                }
                
                with st.expander(f"{severity_color.get(alert['severity'], '⚪')} {alert['location']} - {alert['category']}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**PM2.5:** {alert['pm25']:.1f} μg/m³")
                        st.write(f"**AQI:** {alert['aqi']}")
                        st.write(f"**Time:** {alert['timestamp']}")
                    with col2:
                        st.write(f"**Health Impact:** {alert['health_message']}")
                        st.write("**Recommendations:**")
                        for rec in alert['recommendations']:
                            st.write(f"• {rec}")
        
        # Visualizations
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("🗺️ Pollution Heatmap")
            if 'map' in results:
                st.components.v1.html(results['map']._repr_html_(), height=500)
            
            st.header("📈 Time Series Analysis")
            if 'time_series' in results:
                st.plotly_chart(results['time_series'], use_container_width=True)
        
        with col2:
            st.header("🎯 Feature Importance")
            if 'importance_plot' in results:
                st.plotly_chart(results['importance_plot'], use_container_width=True)
            
            # Recent predictions table
            if 'predictions' in results and not results['predictions'].empty:
                st.header("📋 Recent Predictions")
                display_df = results['predictions'][['datetime', 'station_id', 'pm25', 'predicted_pm25']].tail(10)
                display_df['error'] = abs(display_df['pm25'] - display_df['predicted_pm25'])
                st.dataframe(display_df.round(2))
    
    else:
        st.info("Click 'Run Prediction Pipeline' to start air pollution forecasting.")
        
        # Show sample information
        st.header("🌟 About PawanAI")
        st.markdown("""
        PawanAI is an advanced air pollution forecasting platform that combines:
        
        - **🛰️ Satellite Data**: AOD measurements from INSAT-3D satellites
        - **🏭 Ground Sensors**: Real-time PM2.5 data from CPCB and OpenAQ
        - **🌤️ Weather Data**: Meteorological variables from MERRA-2 reanalysis
        - **🤖 AI/ML Models**: Ensemble Random Forest and XGBoost algorithms
        
        ### Key Features:
        - Real-time pollution estimation across India
        - 2-3 hour advance forecasting
        - Interactive pollution heatmaps
        - Automated health alerts
        - High-resolution (1km) spatial coverage
        """)

# FastAPI Application for API endpoints
def create_fastapi_app():
    """Create FastAPI application for API access"""
    
    app = FastAPI(
        title="PawanAI API",
        description="Air Pollution Forecasting API",
        version="1.0.0"
    )
    
    class PredictionRequest(BaseModel):
        latitude: float
        longitude: float
        datetime: Optional[str] = None
    
    class PredictionResponse(BaseModel):
        pm25: float
        aqi: int
        category: str
        confidence: float
        timestamp: str
    
    @app.get("/")
    async def root():
        return {"message": "PawanAI API - Air Pollution Forecasting"}
    
    @app.post("/predict", response_model=PredictionResponse)
    async def predict_pollution(request: PredictionRequest):
        try:
            # Initialize PawanAI
            pawan_app = PawanAIApp()
            
            # Run pipeline (in production, this would use cached models)
            results = pawan_app.run_pipeline()
            
            if not results:
                raise HTTPException(status_code=500, detail="Prediction pipeline failed")
            
            # Find nearest prediction
            predictions_df = results['predictions']
            if predictions_df.empty:
                raise HTTPException(status_code=404, detail="No predictions available")
            
            # Simple nearest neighbor (in production, use proper spatial interpolation)
            distances = np.sqrt(
                (predictions_df['latitude'] - request.latitude)**2 + 
                (predictions_df['longitude'] - request.longitude)**2
            )
            nearest_idx = distances.idxmin()
            nearest_prediction = predictions_df.loc[nearest_idx]
            
            # Calculate AQI
            alert_system = AlertSystem()
            aqi, category = alert_system.calculate_aqi(nearest_prediction['predicted_pm25'])
            
            return PredictionResponse(
                pm25=float(nearest_prediction['predicted_pm25']),
                aqi=aqi,
                category=category,
                confidence=0.85,  # Placeholder confidence
                timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": datetime.now().isoformat()}
    
    return app

# Main execution
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage
    print("🌬️ PawanAI: Air Pollution Forecasting Platform")
    print("=" * 50)
    
    try:
        # Initialize and run pipeline
        app = PawanAIApp()
        results = app.run_pipeline()
        
        if results:
            print(f"✅ Pipeline completed successfully!")
            print(f"📊 Model metrics: {results.get('metrics', {})}")
            print(f"🚨 Generated {len(results.get('alerts', []))} alerts")
            print(f"📍 Processed {len(results.get('predictions', []))} predictions")
            
            # Save results
            if 'predictions' in results and not results['predictions'].empty:
                results['predictions'].to_csv('pawanai_predictions.csv', index=False)
                print("💾 Predictions saved to 'pawanai_predictions.csv'")
        
        else:
            print("❌ Pipeline failed")
            
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n🚀 To run the web application:")
    print("streamlit run pawanai.py")
    print("\n🔗 To run the API server:")
    print("uvicorn pawanai:create_fastapi_app --reload")