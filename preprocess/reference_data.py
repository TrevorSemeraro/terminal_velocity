import sys
import pandas as pd
import numpy as np
import os

sys.path.append(os.path.dirname(os.getcwd()))

from lib.fast_velocity import beard_terminal_velocity_numba

diameter = np.logspace(np.log10(1e-6), np.log10(7e-3), 100_000)
temperature = np.full_like(diameter, 285)
pressure = np.full_like(diameter, 90_000)
v_t = beard_terminal_velocity_numba(diameter, temperature, pressure)

df = pd.DataFrame({
    'diameter': diameter,
    'temperature': temperature,
    'pressure': pressure,
    'v_t': v_t
})

df.to_csv('../data/reference_data.csv', index=False)