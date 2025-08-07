import sys
import pandas as pd
import numpy as np
import sympy as sp
import os

sys.path.append(os.path.dirname(os.getcwd()))

from lib.sympy_helper import get_function_from_output
from lib.sampler import generate_grid, generate_sample

# INPUTS
output_file = "msre-ref-4"

# Pre Train Data
test_data = generate_sample(num_samples=10_000)
func = get_function_from_output(output_file)

test_data['pred_vt'] = test_data.apply(lambda row: func(row['diameter']), axis=1)
test_data['correction'] = test_data['v_t'] / test_data['pred_vt']

test_data.to_csv(f"../data/correction_data_{output_file}.csv", index=False)

# Fine Tune Data
test_data = generate_grid(size=50)
func = get_function_from_output(output_file)

test_data['pred_vt'] = test_data.apply(lambda row: func(row['diameter']), axis=1)
test_data['correction'] = test_data['v_t'] / test_data['pred_vt']

test_data.to_csv(f"../data/final_correction_data_{output_file}.csv", index=False)