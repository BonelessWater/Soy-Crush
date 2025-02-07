import sqlite3
import csv
import requests
import gzip
import io

# Connect to the SQLite database (data.db)
conn = sqlite3.connect("data.db")
cursor = conn.cursor()

# Drop the temperatures table if it exists, then create a new one.
cursor.execute("DROP TABLE IF EXISTS temperatures")
cursor.execute("""
CREATE TABLE IF NOT EXISTS temperatures (
    station_id TEXT,
    date TEXT,
    tavg REAL,
    tmin REAL,
    tmax REAL,
    prcp REAL,
    snow REAL,
    wdir REAL,
    wspd REAL,
    wpgt REAL,
    pres REAL,
    tsun REAL,
    PRIMARY KEY (station_id, date)
)
""")
conn.commit()

# Get all station IDs from the stations table.
cursor.execute("SELECT id FROM stations")
station_ids = [row[0] for row in cursor.fetchall()]

# Function to safely convert string values to float.
def safe_float(value):
    try:
        if value is not None:
            value = value.strip()
        return float(value) if value not in ("", None) else None
    except ValueError:
        return None

# Define the proper fieldnames as per Meteostat's documentation.
# Order: 1.date, 2.tavg, 3.tmin, 4.tmax, 5.prcp, 6.snow, 7.wdir,
# 8.wspd, 9.wpgt, 10.pres, 11.tsun
fieldnames = ["date", "tavg", "tmin", "tmax", "prcp", "snow", "wdir", "wspd", "wpgt", "pres", "tsun"]

# Loop over each station ID, download and process its temperature CSV.
for station_id in station_ids:
    # Construct the URL using the .csv.gz extension.
    url = f"https://bulk.meteostat.net/v2/daily/{station_id}.csv.gz"
    print(f"Processing station {station_id} from URL: {url}")
    
    try:
        # Download the gzipped CSV file.
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        
        # Decompress the content using gzip and wrap it in a text stream.
        with gzip.open(io.BytesIO(response.content), mode='rt', encoding='utf-8') as f:
            # Supply the fieldnames so that the first row’s values are not treated as headers.
            reader = csv.DictReader(f, fieldnames=fieldnames)
            temp_rows = []
            
            for row in reader:
                # Extract only the date and temperature columns.
                date = row.get("date")
                tavg = safe_float(row.get("tavg"))
                tmin = safe_float(row.get("tmin"))
                tmax = safe_float(row.get("tmax"))
                prcp = safe_float(row.get("prcp"))
                snow = safe_float(row.get("wdir"))
                wdir = safe_float(row.get("wspd"))
                wpgt = safe_float(row.get("wpgt"))
                pres = safe_float(row.get("pres"))
                tsun = safe_float(row.get("tsun"))
                
                temp_rows.append((station_id, date, tavg, tmin, tmax, prcp, snow, wdir, wpgt, pres, tsun))
            
            if temp_rows:
                cursor.executemany("""
                    INSERT OR REPLACE INTO temperatures (station_id, date, tavg, tmin, tmax, prcp, snow, wdir, wpgt, pres, tsun)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, temp_rows)
                conn.commit()
                print(f"Inserted {len(temp_rows)} temperature records for station {station_id}.")
            else:
                print(f"No temperature records found for station {station_id}.")
    
    except requests.exceptions.HTTPError as http_err:
        # This will catch 404 errors and other HTTP errors.
        print(f"HTTP error for station {station_id}: {http_err}")
    except Exception as e:
        print(f"Error processing station {station_id}: {e}")

conn.close()
print("Temperature data import complete.")
