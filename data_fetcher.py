import json
import sqlite3

# Load the JSON data
with open("stations.json", "r", encoding="utf-8") as file:
    stations = json.load(file)

# Define Midwest US longitude range
MIN_LONGITUDE = -100
MAX_LONGITUDE = -80
COUNTRY = "US"

# Filter Midwest US stations
midwest_stations = [
    station for station in stations 
    if station.get("country") == COUNTRY and 
       MIN_LONGITUDE <= station["location"]["longitude"] <= MAX_LONGITUDE
]

# Connect to SQLite database
conn = sqlite3.connect("data.db")
cursor = conn.cursor()

# Create table for stations
cursor.execute("""
CREATE TABLE IF NOT EXISTS stations (
    id TEXT PRIMARY KEY,
    name TEXT,
    country TEXT,
    region TEXT,
    latitude REAL,
    longitude REAL,
    elevation REAL,
    timezone TEXT
)
""")

# Insert filtered stations
for station in midwest_stations:
    cursor.execute("""
    INSERT INTO stations (id, name, country, region, latitude, longitude, elevation, timezone)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        station["id"],
        station["name"]["en"],
        station["country"],
        station["region"],
        station["location"]["latitude"],
        station["location"]["longitude"],
        station["location"]["elevation"],
        station["timezone"]
    ))

# Commit and close connection
conn.commit()
conn.close()

print(f"Inserted {len(midwest_stations)} stations into data.db")
