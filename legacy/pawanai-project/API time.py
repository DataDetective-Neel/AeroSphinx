from urllib.parse import urlencode
from datetime import datetime, timedelta
import requests, os, time


day = int(input('Enter No of days: '))
stations_data = [
    {"state": "Delhi", "city": "Delhi", "station": "R.K. Puram"},
    {"state": "Delhi", "city": "Delhi", "station": "Anand Vihar"},
    {"state": "Maharashtra", "city": "Mumbai", "station": "Bandra"},
    {"state": "West Bengal", "city": "Kolkata", "station": "Rabindra Bharati University"},
    {"state": "Tamil Nadu", "city": "Chennai", "station": "Alandur Bus Depot"}
]

parameters = ["PM2.5", "PM10"]
start_date = datetime.now() - timedelta(days=day)
end_date = datetime.now()

os.makedirs("downloads", exist_ok=True)

for entry in stations_data:
    for parameter in parameters:
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime("%Y-%m-%d")
            data = urlencode({
                "state": entry["state"],
                "city": entry["city"],
                "station": entry["station"],
                "parameter": parameter,
                "date": date_str
            })

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Referer": "https://airquality.cpcb.gov.in/",
                "User-Agent": "Mozilla/5.0"
            }

            try:
                res = requests.post("https://airquality.cpcb.gov.in/dataRepository/file_Path", data=data, headers=headers, timeout=10)
                fname = f"{entry['station'].replace('.', '').replace(' ', '_').lower()}_{parameter}_{date_str}.csv"
                if b"utf-8" not in res.content[:100]:
                    with open(f"downloads/{fname}", "wb") as f:
                        f.write(res.content)
                    print(f"✅ Saved: {fname}")
                else:
                    print(f"❌ No data for {date_str} - {entry['station']} - {parameter}")
            except Exception as e:
                print(f"💥 Error on {date_str}: {e}")
            current_date += timedelta(days=1)
            time.sleep(0.5)
