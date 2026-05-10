import requests, pandas as pd
from datetime import datetime

def fetch_openaq():
    url = "https://api.openaq.org/v2/latest?country=IN&parameter=pm25&limit=1000"
    resp = requests.get(url, timeout=10).json().get('results',[])
    rows=[]
    for it in resp:
        m=it['measurements'][0]
        rows.append({
            'location':  it['location'],
            'city':      it['city'],
            'latitude':  it['coordinates']['latitude'],
            'longitude': it['coordinates']['longitude'],
            'pm25':      m['value'],
            'unit':      m['unit'],
            'timestamp': m['lastUpdated'],
            'source':    'OpenAQ'
        })
    return pd.DataFrame(rows)

def fetch_cpcb():
    url = "https://app.cpcbccr.com/ccr/GetCityWiseAQI"
    hdr={"Content-Type":"application/json","User-Agent":"Mozilla/5.0"}
    resp = requests.post(url, headers=hdr, timeout=10).json()
    rows=[]
    for city in resp:
        for st in city.get('Stations',[]):
            p=st.get('Parameters',{}).get('PM2.5',{})
            if 'Last_Value' in p:
                rows.append({
                    'location':  st['StationName'],
                    'city':      city['City'],
                    'latitude':  st['Latitude'],
                    'longitude': st['Longitude'],
                    'pm25':      p['Last_Value'],
                    'unit':      'µg/m³',
                    'timestamp': p['Last_Updated'],
                    'source':    'CPCB'
                })
    return pd.DataFrame(rows)

def get_ground_pm25():
    try:
        df=fetch_openaq()
        if df.empty: raise ValueError("Empty OpenAQ")
        return df
    except Exception:
        df2=fetch_cpcb()
        return df2

if __name__=="__main__":
    dfg=get_ground_pm25()
    fname=f"../data/ground_pm25_{datetime.now():%Y%m%d_%H%M}.csv"
    dfg.to_csv(fname,index=False)
    print(f"✅ ground data → {fname}")
