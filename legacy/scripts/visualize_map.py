import folium, pandas as pd

df = pd.read_csv("../data/merged_pm_aod.csv")
m = folium.Map(location=[22.6,79.0], zoom_start=5)

for _,r in df.iterrows():
    folium.CircleMarker(
        [r.latitude,r.longitude],
        radius=5,
        fill=True,
        fill_color='red' if r.pm25>100 else 'green',
        popup=f"{r.location}: {r.pm25} µg/m³"
    ).add_to(m)

m.save("../outputs/pawanai_map.html")
print("✅ map saved")
