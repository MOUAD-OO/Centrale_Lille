import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse

# ---------------------------
# Argument parsing
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("-movement", type=str, help="The movement you want to test", default="test")
args = parser.parse_args()
movement = args.movement

# ---------------------------
# Tags
# ---------------------------
# this part is hard coded !!!! 
tags = [
    "001BC50C70027E04",
    "001BC50C70027E08",
    "001BC50C70027E47",
    "001BC50C70027E5F",
    "001BC50C7002A6B1"
]

# ---------------------------
# Functions
# ---------------------------
def get_json(line):
    return line.split(' ')[1]

def transform(df: pd.DataFrame, freq: float):
    """Normalize timestamps"""
    df['time stamp'] = [i / freq for i in range(len(df))]
    print("Time conversion and normalization are done!")

def save_positions_cartesian(positions, freq, anchors_path='generating_bloc/woltDynDev.json',
                             out_cm='positions_cartesian_cm.npy'):
    """
    Convert list of [lat, lon] -> local cartesian coordinates (meters) using transform in anchors file.
    Saves array in cm.
    """
    try:
        with open(anchors_path, 'r') as f:
            anchors_cfg = json.load(f)
    except Exception:
        anchors_cfg = {}

    t = anchors_cfg.get("transform", {})
    ref_lat = float(t.get("latitude", positions[0][0]))
    ref_lon = float(t.get("longitude", positions[0][1]))
    lon_factor = float(t.get("longitude_factor", 1.0))

    R = 6371000.0  # Earth radius (m)
    ref_lat_rad = np.radians(ref_lat)

    lats = np.radians(np.array([p[0] for p in positions], dtype=float))
    lons = np.radians(np.array([p[1] for p in positions], dtype=float))

    dY = R * (lats - np.radians(ref_lat))
    dX = R * np.cos(ref_lat_rad) * (lons - np.radians(ref_lon)) * lon_factor
    time = np.array([i / freq for i in range(len(positions))])
    arr_m = np.column_stack((dX, dY, time / 100))
    np.save(out_cm,np.array( [arr_m * 100.0]))  # cm
    print(f"Saved {arr_m.shape[0]} positions -> {out_cm} (cm)")

# ---------------------------
# Load logs
# ---------------------------
list_meta_data = []
with open('logs/test.log', 'r') as file:
    for line in file:
        if line.startswith('/applink/'):
            list_meta_data.append(get_json(line))

# ---------------------------
# Load anchors/devices
# ---------------------------
UID = {}
with open('generating_bloc/woltDynDev.json', 'r') as f:
    anchors = json.load(f)
    for device in anchors.get("devices", []):
        UID[device["uid"]] = device.get("altitude", 0)

# ---------------------------
# Initialize data structures
# ---------------------------
data = {tag: [] for tag in tags}
positions = {tag: [] for tag in tags}

# ---------------------------
# Process metadata
# ---------------------------
for line in list_meta_data:
    line = json.loads(line)
    tag = line["meta"]["uid"]

    if tag not in tags:
        continue  # skip unknown tags

    row = {"time stamp": line["timestamp"]}
    for anchor_uid in UID.keys():
        distance = line["poi"]["anchors"].get(anchor_uid, {}).get("dst", 0)
         
        if distance!= 0:
            row[anchor_uid] = np.sqrt(distance ** 2 -  (1 - UID[anchor_uid]) ** 2) * 100  # meters -> cm
        else:
            row[anchor_uid] = distance
    # Append position
    positions[tag].append([float(line["latitude"]), float(line["longitude"])])

    # Append row without the 'tag' key
    data[tag].append(row)

# ---------------------------
# Save CSV for each tag
# ---------------------------
for tag, rows in data.items():
    if not rows:
        continue
    df = pd.DataFrame(rows)
    transform(df, freq=1)
    df.to_csv(f"test_site/{movement}_{tag}.csv", index=False)
    print(f"Saved CSV for tag {tag}")

# ---------------------------
# Save positions as Cartesian arrays
# ---------------------------
for tag, pos in positions.items():
    if not pos:
        continue
    save_positions_cartesian(pos, freq=1,
                             anchors_path='generating_bloc/woltDynDev.json',
                             out_cm=f'test_site/{movement}_{tag}_positions.npy')

        

