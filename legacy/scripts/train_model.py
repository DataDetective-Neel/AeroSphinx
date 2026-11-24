import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("../data/merged_pm_aod.csv", parse_dates=['timestamp'])
# select features + target
X = df[['AOD','latitude','longitude']]  # extend with weather vars later
y = df['pm25']

Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.2,random_state=42)
model = RandomForestRegressor(n_estimators=100,random_state=42)
model.fit(Xtr,ytr)
print("Train R2:", model.score(Xtr,ytr))
print("Test  R2:", model.score(Xte,yte))

joblib.dump(model,"../outputs/pawanai_rf.pkl")
print("✅ model saved")
