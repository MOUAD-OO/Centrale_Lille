import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse
from collections import deque
from time import sleep 
import subprocess
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import time
import asyncio

# ---------------------------
# Argument parsing
# ---------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--movement", type=str, help="The movement you want to test", default="test")
parser.add_argument("--anchor_npy",type=str, help =" The anchors' positions in a np.ndarray ",default="generating_bloc/anchor_positions.npy")
args = parser.parse_args()
movement = args.movement
anchors_array= np.load(args.anchor_npy)

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

def process_metadata(list_meta_data, tags, UID, data, positions):
    """Process metadata logs and populate data and positions dictionaries."""
    for line in list_meta_data:
        line = json.loads(line)
        tag = line["meta"]["uid"]

        if tag not in tags:
            continue  # skip unknown tags

        row = {"time stamp": line["timestamp"]}
        for anchor_uid in UID.keys():
            distance = line["poi"]["anchors"].get(anchor_uid, {}).get("dst", 0)
            
            if distance != 0:
                row[anchor_uid] = np.sqrt(distance ** 2 - (1 - UID[anchor_uid]) ** 2) * 100  # meters -> cm
            else:
                row[anchor_uid] = distance
        
        # Append position
        positions[tag].append([float(line["latitude"]), float(line["longitude"])])

        # Append row
        data[tag].append(row)


async def run_predictor_async(tag, csv_in, npy_out):
    """Run predictor asynchronously and return result."""
    cmd = ["python3", "-m", "data_preparing.predict_path", "--input", csv_in, "--output", npy_out]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            print(f"Predictor finished for {tag}")
            return True
        else:
            print(f"Predictor failed for {tag}: {stderr.decode()}")
            return False
    except Exception as e:
        print(f"Predictor error for {tag}: {e}")
        return False


# ---------------------------
# Initialize data structures
# ---------------------------
data = {tag: [] for tag in tags}
positions = {tag: [] for tag in tags}
pending_tasks = []  # Track async predictor tasks

UID = {}
with open('generating_bloc/woltDynDev.json', 'r') as f:
    anchors = json.load(f)
    for device in anchors.get("devices", []):
        UID[device["uid"]] = device.get("altitude", 0)
        
        
        
# ---------------------------
# Load logs
# ---------------------------
list_meta_data = []
last_meta_data = None




# ---------------------------
# Load logs (real-time tail)
# ---------------------------
file_position = 0
processed_count = 0
 
async def main_loop():
    global file_position, pending_tasks
    
    while True:
        try:
            with open('logs/test.log', 'r') as file:
                loop_start = time.monotonic()
                file.seek(file_position)  # Resume from last position
                new_lines = []
                for line in file:
                    if line.startswith('/applink/'):
                        new_lines.append(get_json(line))
                
                file_position = file.tell()  # Save position for next iteration
                
                if len(new_lines)==0:
                # Wait if no new data
                    await asyncio.sleep(1)
                    continue
                
                
                process_metadata(new_lines, tags, UID, data, positions)
                # Save CSVs and launch predictor asynchronously
                for tag, rows in data.items():
                    if not rows:
                        continue
                    
                    last_hist_len = max(0, len(rows) - 40)
                    rows = rows[last_hist_len:]
                    df = pd.DataFrame(rows)
                    print(len(df))
                    transform(df, freq=1)
                    df.to_csv(f"test_site/{movement}_{tag}.csv", index=False)
                    print(f"Saved CSV for tag {tag}")
                    
                    # Launch predictor asynchronously
                    csv_in = f"test_site/{movement}_{tag}.csv"
                    npy_out = f"test_site/trajectories/{movement}_{tag}.npy"
                    task = asyncio.create_task(run_predictor_async(tag, csv_in, npy_out))
                    pending_tasks.append((tag, task, last_hist_len))
                
                # Clean up completed tasks and update data
                still_pending = []
                for tag, task, last_hist_len in pending_tasks:
                    if task.done():
                        try:
                            task.result()  # Check for exceptions
                            data[tag] = data[tag][last_hist_len:]
                        except Exception as e:
                            print(f"Task error for {tag}: {e}")
                    else:
                        still_pending.append((tag, task, last_hist_len))
                pending_tasks = still_pending
                
                elapsed = time.monotonic() - loop_start
                print(f"Loop took {elapsed:.3f} seconds")
                
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(1)

# Run the async main loop
if __name__ == "__main__":
    asyncio.run(main_loop())