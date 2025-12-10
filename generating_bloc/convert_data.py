import numpy as np
from math import cos, radians
import os
def convert_gps_to_cartesian_cm(ref_data: dict, anchor_data: list):
    file_dir = os.path.dirname(os.path.realpath(__file__))

    EARTH_RADIUS = 6371000.0      # meters
    TO_RAD = np.pi / 180.0

    ref_lat = float(ref_data["latitude"])
    ref_lon = float(ref_data["longitude"])

    # standard longitude scaling
    lon_factor = np.cos(ref_lat * TO_RAD)

    positions_cm = []

    for anchor in anchor_data:
        lat = float(anchor["latitude"])
        lon = float(anchor["longitude"])
        alt = float(anchor["altitude"])

        dy = EARTH_RADIUS * (lat - ref_lat) * TO_RAD
        dx = EARTH_RADIUS * (lon - ref_lon) * TO_RAD * lon_factor
        dz = alt  # already in meters

        positions_cm.append([dx * 100, dy * 100, dz * 100])

    positions_cm = np.array(positions_cm)
    np.save(os.path.join(file_dir, "anchor_positions.npy"), positions_cm)

    return positions_cm

def get_anchor_id(anchor_data:list):
    ids = []
    for i in  range(len(anchor_data)):
        ids.append(anchor_data[i]['uid'])
    return ids



def get_anchors_position_and_id(ref_data: dict, anchor_data:list):
    EARTH_RADIUS = 6371000.0
    TO_RAD = np.pi / 180.0

    ref_lat = float(ref_data['latitude'])
    ref_lon = float(ref_data['longitude'])
        # standard longitude scaling
    lon_factor = np.cos(ref_lat * TO_RAD)

    positions_dict = {}

    for i in range(len(anchor_data)):
        uid = anchor_data[i]['uid']
        lat = float(anchor_data[i]['latitude'])
        lon = float(anchor_data[i]['longitude'])
        alt = float(anchor_data[i]['altitude'])

        dy = EARTH_RADIUS * (lat - ref_lat) * TO_RAD
        dx = EARTH_RADIUS * (lon - ref_lon) * TO_RAD * lon_factor
        dz = alt

        positions_dict[uid] = [dx * 100, dy * 100, dz * 100]  # cm

    return positions_dict
