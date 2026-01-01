import requests
import pandas as pd

def fetch_waterlevel_yavatmal_post(start_date, end_date, page=0, size=10):
    """
    Fetch water level data via POST request for Yavatmal district, Maharashtra.
    """
    url = "https://indiawris.gov.in/Dataset/River Water Level"
    params = {
        "stateName": "Maharashtra",
        "districtName": "Yavatmal",
        "agencyName": "Maharashtra sw",
        "startdate": start_date,
        "enddate": end_date,
        "download": "false",
        "page": page,
        "size": size
    }

    headers = {
        "Accept": "application/json",
        "User-Agent": "Mozilla/5.0"
    }

    try:
        resp = requests.post(url, headers=headers, params=params, data="", timeout=20)
        resp.raise_for_status()

        try:
            data = resp.json()
        except ValueError:
            print("❌ Response not valid JSON:\n", resp.text[:400])
            return pd.DataFrame()

        print("✅ Keys in JSON:", data.keys())
        if 'content' in data:
            df = pd.DataFrame(data['content'])
        elif 'data' in data:
            df = pd.DataFrame(data['data'])
        else:
            print("⚠️ Unexpected JSON structure:", list(data.keys()))
            return pd.DataFrame()
        


        # Convert date column and rename water level
        if 'dataTime' in df.columns:
            df['datatime'] = pd.to_datetime(df['dataTime'])
            df = df.set_index('datatime')


        # if 'stationName' in df.columns:
        #     df = df[['stationName']]
        # else:
        #     print("⚠️ Could not find stationName column:", df.columns)
        #     return pd.DataFrame()
        if 'level' in df.columns:
            df = df[['level']]
        elif 'dataValue' in df.columns and 'stationName' in df.columns:
            df = df.rename(columns={'dataValue': 'level'})[['level', 'stationName']]
        else:
            print("⚠️ Could not find water level and stationName column:", df.columns)
            return pd.DataFrame()
        

        return df,data

    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        print(f"❌ Network / Request error: {e}")
        return pd.DataFrame()


# Example usage
df_yavatmal_post,data_json = fetch_waterlevel_yavatmal_post(
    start_date="2025-10-15",
    end_date="2025-10-25",
    page=0,
    size=22   
)

print(df_yavatmal_post.head(22))
# print(data_json)

