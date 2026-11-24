from urllib.parse import urlencode
import requests
import os

url = "https://airquality.cpcb.gov.in/dataRepository/file_Path"
headers = {
    "Content-Type": "application/x-www-form-urlencoded",
    "Referer": "https://airquality.cpcb.gov.in/",
    "User-Agent": "Mozilla/5.0"
}

data = urlencode({
    "state": "Delhi",
    "city": "Delhi",
    "station": "R.K. Puram",
    "parameter": "PM2.5",
    "date": "2025-07-04"
})

response = requests.post(url, data=data, headers=headers)

# Save binary content directly
filename = "rkpuram_pm25_20250704.csv"
with open(filename, "wb") as f:
    f.write(response.content)

print(f"✅ File saved as: {filename}")
