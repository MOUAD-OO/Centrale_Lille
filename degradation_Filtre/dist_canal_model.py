#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import argparse
import os
import pandas as pd
import configparser

config=configparser.ConfigParser()

dir_path = os.path.dirname(os.path.realpath(__file__))

config.read("generator_config.ini")

matrix_section = "CANAL_MODEL"
default_dist_matrix = config.get(matrix_section, "Canal_input", fallback="generating_bloc/dist_matrix.csv")
default_output = config.get(matrix_section, "output", fallback="degraded_dist_matrix.csv")
default_max_mask_ratio = config.getfloat(matrix_section, "max_mask_ratio", fallback=0.3)
default_noise_level = config.getfloat(matrix_section, "noise_level")

parser = argparse.ArgumentParser(
    description="Degrade path data by adding noise and masking points.",
    usage="python %(prog)s --dist_matrix <input_file> --output <output_file> [--max_mask_ratio <ratio>] [--noise_level <level>]"
)
parser.add_argument(
    "--dist_matrix",
    type=str,
    default=default_dist_matrix,
    help="Path to the input .csv file containing path data."
)
parser.add_argument(
    "--output",
    type=str,
    default=default_output,
    help="Path to save the degraded .csv file."
)
parser.add_argument(
    "--max_mask_ratio",
    type=float,
    default=default_max_mask_ratio,
    help="Maximum ratio of points to mask in each path."
)
parser.add_argument(
    "--noise_level",
    type=float,
    default=default_noise_level,
    help="Standard deviation of the Gaussian noise to add to the path coordinates."
)

args = parser.parse_args()

max_mask_ratio = args.max_mask_ratio
noise_level = args.noise_level
input_matrix= args.dist_matrix
output_matrix = args.output

#type verification
if not input_matrix.endswith('.csv'):
    raise ValueError("Input file must be a .csv file")
if not output_matrix.endswith('.csv'):
    raise ValueError("Output file must be a .csv file")

def Mask_data(matrix , max_mask_ratio: float):
    
    masked_matrix = matrix.copy()
    masked_matrix.drop(columns=["time stamp"],inplace=True)
    mask_ratio=np.random.uniform(0,max_mask_ratio)
    
    for column in masked_matrix.columns:
        mask=np.random.rand(len(matrix[column])) < max_mask_ratio
        masked_matrix.loc[mask,column]=0 # Using inf to indicate missing data
    
    masked_matrix["time stamp"]=matrix["time stamp"]  
    columns=masked_matrix.columns
    columns=list(columns)
    columns[0],columns[-1]=columns[-1],columns[0]
    masked_matrix=masked_matrix[columns]
    return masked_matrix

def noise_data_normal(matrix, noise_level: float) -> pd.DataFrame:
    noisy_matrix = matrix.copy()
    noisy_matrix.drop(columns=["time stamp"], inplace=True)

    shape = noisy_matrix.shape
    # zero-mean Gaussian additive noise
    if noise_level == 0:
        noise = np.zeros(shape)
    else:
        noise = np.random.normal(loc=0.0, scale=noise_level, size=shape)

    # Apply noise only to non-zero entries (missing data encoded as 0)
    mask_nonzero = (noisy_matrix != 0).values
    values = noisy_matrix.values.astype(float)
    values[mask_nonzero] += noise[mask_nonzero]

    # Clip invalid results
    values[values < 0] = 0          # distances cannot be negative
    values[values > 2500] = 0       # existing business rule: >2500 → mask as 0

    noisy_matrix = pd.DataFrame(values, columns=noisy_matrix.columns, index=noisy_matrix.index)
    noisy_matrix["time stamp"] = matrix["time stamp"]
    return noisy_matrix

def degrade_data(dist_matrix: np.ndarray, max_mask_ratio: float, noise_level: float) -> np.ndarray:
    noisy_matrix = noise_data_normal(dist_matrix, noise_level)
    degraded_matrix = Mask_data(noisy_matrix, max_mask_ratio)
    return degraded_matrix


def main():

    dist_matrix=pd.read_csv(input_matrix,sep=",")
    degraded_matrix= degrade_data(dist_matrix, max_mask_ratio, noise_level)
    degraded_matrix.to_csv(output_matrix,index=False)
    print(f"Degraded distance matrix saved to {output_matrix}")
    
    
if __name__ == "__main__":
    main()