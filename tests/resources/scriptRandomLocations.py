import json
import random
from datetime import datetime, timedelta

TARGET_TOTAL = 20000

def random_date(start_year=2000, end_year=2025):
    start = datetime(start_year, 1, 1)
    end = datetime(end_year, 12, 31)
    delta = end - start
    random_seconds = random.randint(0, int(delta.total_seconds()))
    return (start + timedelta(seconds=random_seconds)).strftime("%Y-%m-%dT%H:%M")

def random_latitude():
    return round(random.uniform(-90.0, 90.0), 6)

def random_longitude():
    return round(random.uniform(-180.0, 180.0), 6)

with open("locations_1000_full.json", "r", encoding="utf-8") as f:
    locations = json.load(f)

current_count = len(locations)
to_add = TARGET_TOTAL - current_count

if to_add <= 0:
    print("Already have 20,000 or more locations. No new data added.")
else:
    for _ in range(to_add):
        locations.append({
            "eventDate": random_date(),
            "decimalLatitude": random_latitude(),
            "decimalLongitude": random_longitude()
        })

    with open("locations_20000.json", "w", encoding="utf-8") as f:
        json.dump(locations, f, indent=2)

    print(f"Added {to_add} locations. Total = {len(locations)}")
