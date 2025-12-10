# Dynamic_Loc Path Data Generator

This script generates random paths (linear, polynomial, circular, and sinusoidal) scaled to your anchor positions and saves them as `.npy` files. It can also plot the generated paths and anchor positions.

---

## Requirements

- Python 3.x
- numpy
- matplotlib
- argparse

You also need a JSON file with anchor positions (see `test_anchor.json` for format).

---

## Usage

Run the script from the command line:

```sh
python data_generator.py [--Anchor_position FILE] [--number_of_paths N] [--num_dots N] [--with_plots] [--plot_all_in_one] [--mean_speed FLOAT] [--freq FLOAT]
```

### Arguments

- `--Anchor_position FILE`  
  Path to the anchor position JSON file.  
  **Default:** `/home/nathan/Dynamic_Loc/test_anchor.json`

- `--number_of_paths N`  
  Number of paths to generate.  
  **Default:** `5`

- `--num_dots N`  
  Number of points per path.  
  **Default:** calculated from mean speed and frequency, or `10` if not specified.

- `--with_plots`  
  If set, plots each generated path type with anchor positions.

- `--plot_all_in_one`  
  If set, plots all generated paths from all files in one figure.

- `--mean_speed FLOAT`  
  Mean speed of the moving object in m/s (used for auto-calculating number of points).  
  **Default:** `1.3`

- `--freq FLOAT`  
  Frequency of measurements in Hz (used for auto-calculating number of points).  
  **Default:** `1.0`

---

### Example

Generate 20 paths with 30 points each and plot all in one figure:

```sh
python data_generator.py --Anchor_position my_anchors.json --number_of_paths 20 --num_dots 30 --plot_all_in_one
```

Generate paths and plot each type separately:

```sh
python data_generator.py --with_plots
```

Use a custom anchor file and default settings:

```sh
python data_generator.py --Anchor_position /path/to/anchors.json
```

---

## Output

- Generated paths are saved as `.npy` files in the `paths/` directory:
  - `linear_paths.npy`
  - `polynomial_paths.npy`
  - `circular_paths.npy`
  - `wave_paths.npy`

- Plots are shown if `--with_plots` or `--plot_all_in_one` is used.

---

## Anchor JSON Format

Your anchor JSON should look like:

```json
{
  "transform": {
    "latitude": ...,
    "longitude": ...,
    "altitude": ...,
    "longitude_factor": ...
  },
  "devices": [
    {"latitude": ..., "longitude": ..., "altitude": ...},
    ...
  ]
}
```

---

## Notes

- The script will create the `paths/` directory if it does not exist.
- All generated paths are scaled to the anchor positions for realistic simulation.

---

**Author:** Nathan
