#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import os

folder_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def speed(A, B, frequency):
    """Compute speed between two points A and B given the frequency."""
    A = np.asarray(A)
    B = np.asarray(B)
    if A.shape[0] < 2 or B.shape[0] < 2:
        return 0
    distance = np.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)
    return distance * frequency / 100  # m/s

def acceleration(position , path , frequency):
    """Compute acceleration at a given position in the path."""
    if position == 0 or position == len(path) - 1:
        return 0  # No acceleration at the start or end of the path
    v1 = speed(path[position - 1][:2], path[position][:2], frequency)
    v2 = speed(path[position][:2], path[position + 1][:2], frequency)
    return (v2 - v1) * frequency # m/s^2




def speed_time_series(path, frequency):
    """Compute speed time series for the entire path."""
    return [speed(path[i - 1], path[i], frequency) if i > 0 else 0 for i in range(len(path))]

def acceleration_time_series(path, frequency):
    """Compute acceleration time series for the entire path."""
    return [acceleration(i, path, frequency) for i in range(len(path))]


def linear_speed_time_series(path, frequency):
    """Compute linear speed time series for the entire path."""
    return [
        np.linalg.norm(path[i - 1] - path[i]) * frequency / 100 if i > 0 else 0
        for i in range(len(path))
    ]
    
### NEED OPTIMIZATION ###
def velocity_vector_time_series(path, frequency):
    """Compute velocity vector time series for the entire path."""
    velocity_vectors = []
    for i in range(len(path)):
        if i == 0:
            velocity_vectors.append(np.array([0, 0]))  # No velocity at the first point
        else:
            velocity = (path[i][:2] - path[i - 1][:2]) * frequency / 100
            velocity_vectors.append(velocity)
    return np.array(velocity_vectors)
    
def acceleration_vector_time_series(path, frequency):
    """Compute acceleration vector time series for the entire path."""
    velocity_vectors = velocity_vector_time_series(path, frequency)
    acceleration_vectors = []
    for i in range(len(velocity_vectors)):
        if i == 0:
            acceleration_vectors.append(np.array([0, 0]))  # No acceleration at the first point
        else:
            acceleration = (velocity_vectors[i] - velocity_vectors[i - 1]) * frequency
            acceleration_vectors.append(acceleration)
    return np.array(acceleration_vectors)