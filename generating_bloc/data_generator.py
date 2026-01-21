#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt 
import argparse
import json
from convert_data import convert_gps_to_cartesian_cm
import configparser

dir_path = os.path.dirname(os.path.realpath(__file__))
config = configparser.ConfigParser()
config.read("generator_config.ini")

# Get defaults from config file, with fallback values
default_anchor = config.get("GENERATE", "Anchor_position", fallback=os.path.join(dir_path, "woltDynDev.json"))
default_num_paths = config.getint("GENERATE", "number_of_paths", fallback=5)
default_mean_speed = config.getfloat("GENERATE", "mean_speed", fallback=1.3)
default_freq = config.getfloat("GENERATE", "freq", fallback=1.0)
default_with_plots = config.getboolean("GENERATE", "with_plots", fallback=False)
default_plot_all = config.getboolean("GENERATE", "plot_all_in_one", fallback=False)
seed=config.getint("GENERATE","seed")

parser = argparse.ArgumentParser(
    description=(
        "This script generates random paths (graphs) and saves them as .npy files.\n"
        "By default, it will generate 5 paths of each type (linear, polynomial, circular, sinusoidal).\n"
        "Default parameters:\n"
        "  --Anchor_position: ./test_anchor.json\n"
        "  --number_of_paths: 5\n"
        "  --with_plots: False\n"
        "  --plot_all_in_one: False\n"
        "  --mean_speed: 1.3 (m/s)\n"
        "  --freq: 1.0 (Hz)\n"
    ),
    usage="python data_generator.py [--Anchor_position JSON FILE] [--number_of_paths N] [--with_plots] [--plot_all_in_one] [--mean_speed SPEED] [--freq FREQ]"
)

parser.add_argument("--Anchor_position", help="Path to the anchor position JSON file", type=str, default=default_anchor)
parser.add_argument("-n","--number_of_paths", help="Number of paths to generate", type=int, default=default_num_paths)
parser.add_argument("--with_plots", help="Plot the generated paths", action='store_true', default=default_with_plots)
parser.add_argument("--plot_all_in_one", help="Plot all generated paths in one figure", action='store_true', default=default_plot_all)
parser.add_argument("--mean_speed", help="Mean speed of the moving object (m/s)", type=float, default=default_mean_speed)
parser.add_argument("--freq", help="Frequency of measurements (Hz)", type=float, default=default_freq)





args = parser.parse_args()


anchor_position_file=args.Anchor_position #path to the anchor position file (json format)
number_of_paths=args.number_of_paths #number of paths (graphs)
with_plots=args.with_plots #if you want to plot the generated paths
plot_all_in_one=args.plot_all_in_one
mean_speed=args.mean_speed #mean speed of the moving object (m/s)
freq=args.freq #frequency of measurements (Hz)





    


def anchor_pos(anchor_position_file):
    """ Get the anchor positions in cartesian coordinates (cm) from a json file.
    Args:
        anchor_position_file : str
            The file path to the anchor position (json format).

    Returns:
    anchor_position: np.ndarray
        The anchor positions in cartesian coordinates (cm).  
    """ 
    print("------------------------------------------------------------------")
    print("extracting anchors' position...")
    print("------------------------------------------------------------------")
    with open(anchor_position_file, "r") as f:
        anchor_position = json.load(f)
        
    devices=anchor_position["devices"]
    anchor_position= convert_gps_to_cartesian_cm(anchor_position["transform"], devices)
    print("extracting is done !")
    return anchor_position



#getr the anchor position in cartesian coordinates 
anchor_position = anchor_pos(anchor_position_file)



