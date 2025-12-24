import pandas as pd
import numpy as np
import json
from time import sleep
import argparse
from collections import deque
import asyncio
from data_preparing.predict_path_inc import PositionEstimator
from data_preparing.clean_positions import noise_cleaning
import sys

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
# Initialize data structures
# ---------------------------
data = {tag: [] for tag in tags}
trajectories = {tag: np.empty((1,1,3)) for tag in tags}
positions = {tag: [] for tag in tags}
pending_tasks = []

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
                try:
                    row[anchor_uid] = np.sqrt(distance ** 2 - (0 - UID[anchor_uid]) ** 2) * 100  # meters -> cm
                except ValueError:
                    row[anchor_uid] = distance*100
            else:
                row[anchor_uid] = distance*100
        
        # Append position
        positions[tag].append([float(line["latitude"]), float(line["longitude"])])
        save_positions_cartesian(positions[tag], 1,out_cm=f"{tag}positions_cartesian_cm.npy")
        # Append row
        data[tag].append(row)

def intersection_cercles(x0, y0, r0, x1, y1, r1):
    d = np.hypot(x1 - x0, y1 - y0)

    # Aucun ou infini de points
    if d > r0 + r1 or d < abs(r0 - r1) or (d == 0 and r0 == r1):
        return None

    # Calcul intermédiaire
    a = (r0**2 - r1**2 + d**2) / (2 * d)
    h = np.sqrt(r0**2 - a**2)

    # Point P2 sur la ligne entre les centres
    x2 = x0 + a * (x1 - x0) / d
    y2 = y0 + a * (y1 - y0) / d

    # Points d’intersection
    rx = -(y1 - y0) * (h / d)
    ry =  (x1 - x0) * (h / d)

    p1 = (x2 + rx, y2 + ry)
    p2 = (x2 - rx, y2 - ry)
    
    return [p1, p2]

def kalman_filter(positions, time_window = 3, coeff=0.25, freq=1):
    """
    Apply a simplified Kalman-like filter on the entire trajectory.

    Args:
        positions (np.ndarray): array of positions, shape (n, d)
        time_window (int): number of steps to look back for velocity estimation
        coeff (float): blending coefficient (0..1)
        freq (float): measurement frequency in Hz

    Returns:
        np.ndarray: filtered trajectory, same shape as positions
    """
    n, d = positions.shape
    d=d-1
    dt = 1.0 / freq

    # Copy positions to initialize filtered trajectory
    filtered = np.copy(positions)
    old_velocity = np.zeros(d)
    for i in range(2, n):
        # Determine the start of the window
        start_idx = max(0, i - time_window)
        pos_start = filtered[start_idx,:2]
        pos_n_1 = filtered[i - 1,:2]

        # Compute velocity from start of window to previous point
        old_dt = (i - 1 - start_idx) * dt
        if old_dt == 0:
            old_velocity = np.zeros(d)
        else:
            v = (pos_n_1 - pos_start) / old_dt
            if np.linalg.norm(v) >300:
                old_velocity = v #cap velocity 

        # Extrapolate to current step
        new_dt = dt
        estimated_pos = pos_n_1 + old_velocity * new_dt

        # Blend with the current measurement
        filtered[i,:2] = coeff * estimated_pos + (1 - coeff) * positions[i,:2]


    return np.array(filtered)

def prediction(df,trajectory):
    estimator=PositionEstimator()
    
    distance = df
    anchors_path = "generating_bloc/woltDynDev.json" #Hardcoded for now
    step = len(trajectory)-1
    new_path = list(trajectory[0] )
    while step < len(distance):

        try:
            estimated_position = estimator.estimate_position(distance, anchors_path, step)
   
            new_path.append([estimated_position[0],estimated_position[1],step*(1/estimator.freq)])
            if estimator.STATUS[0]=="1_DATA" :
                kalman_pos=kalman_filter(np.array(new_path) ,coeff=0.7,time_window=5)[step,:2]
            
            if estimator.STATUS[0] == "2_DATA" :
                anchors=estimator.STATUS[1]

                
                dots = intersection_cercles(anchors[0][0][0],anchors[0][0][1],anchors[0][1],
                                            anchors[1][0][0],anchors[1][0][1],anchors[1][1])

                kalman_pos=kalman_filter(np.array(new_path) ,coeff=1)[step,:2]
                    # Safe handling when circles do not intersect (intersection_cercles returns None)
                if dots is None:
                    # fallback to Kalman prediction
                    new_path[step][0] = kalman_pos[0]
                    new_path[step][1] = kalman_pos[1]
           
                    print("Warning: circles do not intersect — using Kalman prediction")
                else:
                    p1, p2 = dots
                    # choose the intersection point closest to Kalman prediction
                    if np.linalg.norm(kalman_pos - np.array(p1)) <= np.linalg.norm(kalman_pos - np.array(p2)):
                        chosen = p1
                    else:
                        chosen = p2
                    new_path[step][0] = chosen[0]
                    new_path[step][1] = chosen[1]

                    print("Warning : Only two anchors are detected. Taking the closest position to Kalman prediction ")
               
            if estimator.STATUS[0]== "0_DATA" :
                kalman_pos=kalman_filter(np.array(new_path) ,coeff=0.9,time_window=7)[step,:2]
                new_path[step][0] = kalman_pos[0]
                new_path[step][1] = kalman_pos[1]
                print("Warning : No anchor is detected. Taking position as the Kalman prediction ")
            
            if estimator.STATUS[0]== "OPTIMIZATION_FAIL" :
                new_position = estimator.estimate_position(distance, anchors_path, step)
                new_path[step]=[new_position[0],new_position[1],step*(1/estimator.freq)]
                pass
    
            step+=1
        except ValueError as e:
            print("ValueError :", e)
            df.to_csv(f"error_distance_{step}.csv",index=False)
            print("step =", step)
            break    
        
    new_path = np.array(new_path) 
    raw_path=new_path.copy()
    new_path=noise_cleaning(new_path, window_size=3)

    filtered_path= kalman_filter(new_path[0])

    filtered_path=np.array([filtered_path])
    raw_path=np.array([raw_path])
    #np.save(estimator.output, filtered_path)      
    return  filtered_path,raw_path

def run_predictor(tag,rows,trajectories,movement=movement):
    global last_hist_len
    print("Running run_predictor")
    
    #rows = rows[last_hist_len:]
    last_hist_len = max(0, len(trajectories[tag][0])-1)
    df = pd.DataFrame(rows)
    print("Carried DATA size :",tag,len(df))
    transform(df,freq=1)
    filtered_trajectory,trajectories[tag]= prediction(df,trajectories[tag])
    np.save(f"test_site/trajectories/{movement}_{tag}.npy", filtered_trajectory)

    df.to_csv(f"test_site/{movement}_{tag}_distance.csv",index=False)
    
UID = {}
with open('generating_bloc/woltDynDev.json', 'r') as f:
    anchors = json.load(f)
    for device in anchors.get("devices", []):
        UID[device["uid"]] = device.get("altitude", 0)

# ---------------------------
# Load logs (real-time tail)
# ---------------------------
file_position = 0
last_hist_len=0

while True:
    try:    
        with open('logs/test.log', 'r') as file:
            
            file.seek(file_position)  # Resume from last position
            new_lines = []
            for line in file:
                if line.startswith('/applink/'):
                    new_lines.append(get_json(line))
                        
            file_position = file.tell()
            if len(new_lines)==0:
                sleep(2)
                continue
            process_metadata(new_lines, tags, UID, data, positions)
            for tag, rows in data.items():
                if rows:
                    run_predictor(tag,rows ,trajectories)
                        
                    data = {tag: [] for tag in tags}

                
    except Exception as e:
        print(f"Error: {e}")
        sleep(2)

            

print("Stopping... cleaning up")

#Save everything
for tag, traj in trajectories.items():
    np.save(f"test_site/trajectories/final_{tag}.npy", traj)

#Close resources / flush logs
print("Cleanup done. Exiting.")