import sys
import pandas as pd
import numpy as np
import os

sys.path.append(os.path.dirname(os.getcwd()))

from lib.fast_velocity import beard_terminal_velocity_numba
from lib.sampler import generate_sample

TEST_SAMPLES = 1_000_000
test_data = generate_sample(num_samples=1_000_000, seed=12345)
test_data.to_csv("../data/test_data.csv", index=False, header=False)