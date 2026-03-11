"""
Fetch daily weather data from Open-Meteo Archive API for 10 US East Coast/South cities,
2017-01-01 to 2022-12-31, and save to CSV.
"""

import csv
import json
import os
import time
import urllib.request
import urllib.parse

CITIES = [
    {"name": "New York",      "lat": 40.71, "lon": -74.01, "pop": 20140470},
    {"name": "Philadelphia",  "lat": 39.95, "lon": -75.17, "pop": 6245051},
    {"name": "Miami",         "lat": 25.76, "lon": -80.19, "pop": 6138333},
    {"name": "Atlanta",       "lat": 33.75, "lon": -84.39, "pop": 6089815},
    {"name": "Washington DC", "lat": 38.91, "lon": -77.04, "pop": 6385162},
    {"name": "Boston",        "lat": 42.36, "lon": -71.06, "pop": 4941632},
    {"name": "Charlotte",     "lat": 35.23, "lon": -80.84, "pop": 2660329},
    {"name": "Tampa",         "lat": 27.95, "lon": -82.46, "pop": 3175275},
    {"name": "Orlando",       "lat": 28.54, "lon": -81.38, "pop": 2673376},
    {"name": "Pittsburgh",    "lat": 40.44, "lon": -79.99, "pop": 2370930},
]

API_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_PARAMS = "precipitation_sum,windspeed_10m_max,temperature_2m_max,temperature_2m_min"
YEARS = range(2017, 2023)
OUTPUT_PATH = "/home/user/music-fatalities-update/output/weather_east_south.csv"

COLUMNS = [
    "date", "city", "latitude", "longitude", "population",
    "precipitation_sum", "windspeed_10m_max", "temperature_2m_max", "temperature_2m_min",
]


def fetch_year(city, year):
    """Fetch one year of daily weather for a city, bypassing proxy."""
    params = urllib.parse.urlencode({
        "latitude": city["lat"],
        "longitude": city["lon"],
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "daily": DAILY_PARAMS,
        "timezone": "America/New_York",
    })
    url = f"{API_URL}?{params}"
    # Create a proxy handler that bypasses the proxy for this host
    proxy_handler = urllib.request.ProxyHandler({})
    opener = urllib.request.build_opener(proxy_handler)
    req = urllib.request.Request(url, headers={"User-Agent": "weather-fetch-script/1.0"})
    with opener.open(req, timeout=60) as resp:
        data = json.loads(resp.read().decode())
    return data["daily"]


def main():
    all_rows = []
    total_requests = len(CITIES) * len(list(YEARS))
    done = 0

    for city in CITIES:
        for year in YEARS:
            done += 1
            print(f"[{done}/{total_requests}] Fetching {city['name']} {year}...", flush=True)
            retries = 3
            for attempt in range(retries):
                try:
                    daily = fetch_year(city, year)
                    break
                except Exception as e:
                    print(f"  Attempt {attempt+1} ERROR: {e}", flush=True)
                    if attempt < retries - 1:
                        time.sleep(2)
            else:
                print(f"  SKIPPED {city['name']} {year} after {retries} failures", flush=True)
                continue

            dates = daily["time"]
            precip = daily["precipitation_sum"]
            wind = daily["windspeed_10m_max"]
            tmax = daily["temperature_2m_max"]
            tmin = daily["temperature_2m_min"]

            print(f"  -> {len(dates)} days", flush=True)

            for i, d in enumerate(dates):
                all_rows.append({
                    "date": d,
                    "city": city["name"],
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "population": city["pop"],
                    "precipitation_sum": precip[i],
                    "windspeed_10m_max": wind[i],
                    "temperature_2m_max": tmax[i],
                    "temperature_2m_min": tmin[i],
                })

            time.sleep(0.5)

    # Write CSV
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summary stats
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total rows: {len(all_rows)}")
    if all_rows:
        dates_all = [r["date"] for r in all_rows]
        print(f"Date range: {min(dates_all)} to {max(dates_all)}")
        cities_present = set(r["city"] for r in all_rows)
        print(f"Cities: {len(cities_present)} ({', '.join(sorted(cities_present))})")

        # Check missing data
        weather_cols = ["precipitation_sum", "windspeed_10m_max", "temperature_2m_max", "temperature_2m_min"]
        for col in weather_cols:
            missing = sum(1 for r in all_rows if r[col] is None)
            pct = 100 * missing / len(all_rows)
            print(f"Missing {col}: {missing} / {len(all_rows)} ({pct:.2f}%)")

    print(f"\nSaved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
