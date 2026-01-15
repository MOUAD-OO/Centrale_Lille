#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
import configparser
import os
from data_preparing.physical_atribute import (
    acceleration_time_series,
    speed_time_series
)

config= configparser.ConfigParser()
config.read('generator_config.ini')
freq= config.getfloat("GENERATE", "freq")

folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser(description="this file is used to clean the noise from the path resuting from the localisation algorithm")


parser.add_argument("--input", help="file directory must be .npy format")
parser.add_argument("--output", help="The directory of the claned path")




def noise_cleaning(paths: np.ndarray, window_size: int = 3, freq=freq) -> np.ndarray:
    """
    Clean the noise from the path with two filters:
    1. Outlier suppression: if point i is much larger than the mean of (i-1, i+1),
       replace it by a weighted mean (70% neighbor mean, 30% current point).
    2. Adaptive exponential moving average smoothing.
    
    :param paths: np.ndarray of shape (m, n, 3) or (n, 3) where n is the number of points.
    :param window_size: int, the size of the window used for local exponential smoothing.
    :param freq: int, sampling frequency (Hz), used for computing speed and acceleration.
    :return: np.ndarray of same shape as input, with [x_filtered, y_filtered, time].
    """
    if freq is None :
        raise ValueError("Frequency 'freq' must be provided")
    
    cleaned_paths = []

    # Handle case where input is a single path
    if paths.ndim == 2:
        paths = [paths]
        
    for path in paths:
        filtered_path = path.copy()
        n = path.shape[0]

        # --- Step 1: Outlier suppression ---
        for j in range(1, n - 1):  # skip first and last
            prev_point = filtered_path[j - 1, :2]
            next_point = filtered_path[j + 1, :2]
            current_point = filtered_path[j, :2]

            mean_neighbors = 0.5 * (prev_point + next_point)
            dist_current = np.linalg.norm(current_point - mean_neighbors)

            # Threshold relative to neighbor distance
            if dist_current > 2.5 * np.linalg.norm(prev_point - next_point):
                filtered_path[j, :2] = 0.8 * mean_neighbors + 0.2 * current_point

        # --- Step 2: Adaptive exponential smoothing ---
        for j in range(n):
            start = max(1, j - window_size // 2)
            end = min(n, j + window_size // 2 + 1)
            window = filtered_path[start:end, :2]

            alpha = 0.3
            k = len(window)
            center_idx = j - start
            distances = np.abs(np.arange(k) - center_idx)

            # Compute local dynamics
            acceleration = acceleration_time_series(window, freq)
            speed = speed_time_series(window, freq)
            spd_before = np.abs(np.mean(speed_time_series(filtered_path, freq)))

            acc = np.abs(np.mean(acceleration))
            spd = np.abs(np.mean(speed))

            # Adaptive alpha
            if acc > 6 or spd > 2 * spd_before:
                alpha = 0.002
                distances = np.abs(np.arange(k) - center_idx - 1)
            elif acc > 6 or spd > 3 * spd_before:
                alpha = 0.005
                distances = np.abs(np.arange(k) - center_idx - 1)
            elif acc > 4 or spd > 2 * spd_before:
                alpha = 0.05
                distances = np.abs(np.arange(k) - center_idx - 1)
            elif acc > 3 or spd > 1.3*spd_before:
                alpha = 0.1
                distances = np.abs(np.arange(k) - center_idx - 1)

            # Exponential weights
            weights = (1 - alpha) ** distances
            weights /= weights.sum()

            filtered_path[j, 0] = np.dot(window[:, 0], weights)
            filtered_path[j, 1] = np.dot(window[:, 1], weights)

        # --- Add time column ---
        time = np.arange(n) / freq
        filtered_path = np.column_stack((filtered_path[:, :2], time))
        cleaned_paths.append(filtered_path)

    return np.array(cleaned_paths)


    
   