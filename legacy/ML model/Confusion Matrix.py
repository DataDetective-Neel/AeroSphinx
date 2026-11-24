import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt




# Read files with correct encoding for special characters
df_true = pd.read_csv('pm25_cpcb_05AM_01janto31dec_2019.csv', encoding='latin1')
df_pred = pd.read_csv('predicted_pm25_jan_to_dec_2019.csv')

# Merge by Timestamp and drop missing values
df = pd.merge(
    df_true[['Timestamp', 'PM2.5 (µg/m³)']].dropna(),
    df_pred[['Timestamp', 'PM2.5 (µg/m³)']].dropna(),
    on='Timestamp'
)
print(df.columns.tolist())

def pm25_category(val):
    if pd.isnull(val): return 'Missing'
    val = float(val)
    if val <= 30: return 'Good'
    elif val <= 60: return 'Satisfactory'
    elif val <= 90: return 'Moderate'
    elif val <= 120: return 'Poor'
    elif val <= 250: return 'Very Poor'
    else: return 'Severe'

df['actual_cat'] = df['PM2.5 (µg/m³)_x'].apply(pm25_category)
df['pred_cat'] = df['PM2.5 (µg/m³)_y'].apply(pm25_category)

mask = (df['actual_cat'] != 'Missing') & (df['pred_cat'] != 'Missing')
df = df[mask]
CLASSES = ['Good','Satisfactory','Moderate','Poor','Very Poor','Severe']

cm = confusion_matrix(df['actual_cat'], df['pred_cat'], labels=CLASSES)
plt.figure(figsize=(8,6))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES).plot(cmap='Blues', values_format='d')
plt.title('PM2.5 AQI Category Confusion Matrix')
plt.tight_layout()
plt.savefig('pm25_confusion_matrix.png', dpi=150)
plt.close()
