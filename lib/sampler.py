import numpy as np
import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.getcwd()))
from lib.fast_velocity import beard_terminal_velocity_numba, R_air, calc_dynamic_viscosity
from scipy.stats import qmc

num_inputs = 3

# Define input ranges
diameter_range = (1e-6, 7e-3)  # meters

pressure_range = (6e4, 1.02e5)  # Pa
temperature_range = (230, 310)  # Kelvin

num_inputs = 3

log_diameter_range = np.log(diameter_range)
log_pressure_range = np.log(pressure_range)

lower_bounds = [log_diameter_range[0], temperature_range[0], log_pressure_range[0]]
upper_bounds = [log_diameter_range[1], temperature_range[1], log_pressure_range[1]]

def generate_sample(num_samples=1_000, seed=None):
    """Generate a sample dataset with optional seed for reproducibility."""
    sampler = qmc.LatinHypercube(d=num_inputs, seed=seed)
    sample = sampler.random(n=num_samples)
    sample_scaled = qmc.scale(sample, lower_bounds, upper_bounds)
    
    diameter = np.exp(sample_scaled[:, 0])
    temperature = sample_scaled[:, 1]
    pressure = np.exp(sample_scaled[:, 2])

    v_t = beard_terminal_velocity_numba(
        diameter,
        temperature,
        pressure,
    )

    density = pressure / temperature / R_air
    
    dynamic_viscosity = calc_dynamic_viscosity(temperature)
    
    return pd.DataFrame({
        'diameter': diameter,
        'temperature': temperature,
        'pressure': pressure,
        'v_t': v_t,
        'density': density,
        'dynamic_viscosity': dynamic_viscosity
    })

def generate_grid(size=50):
    diameter_values = np.linspace(log_diameter_range[0], log_diameter_range[1], size)
    pressure_values = np.linspace(log_pressure_range[0], log_pressure_range[1], size)
    temperature_values = np.linspace(220, 320, size)
    
    D, T, P = np.meshgrid(diameter_values, temperature_values, pressure_values)
    
    diameter = np.exp(D.flatten())
    temperature = T.flatten()
    pressure = np.exp(P.flatten())
    
    v_t = beard_terminal_velocity_numba(
        diameter,
        temperature,
        pressure,
    )
    
    density = pressure / temperature / R_air
    dynamic_viscosity = calc_dynamic_viscosity(temperature)

    return pd.DataFrame({
        'diameter': diameter,
        'temperature': temperature,
        'pressure': pressure,
        'v_t': v_t,
        'density': density,
        'dynamic_viscosity': dynamic_viscosity
    })

def generate_grid_samples(size=50, pressure_constant=101_325):
    diameter_values = np.linspace(log_diameter_range[0], log_diameter_range[1], size)
    temperature_values = np.linspace(220, 320, size)
    D, T = np.meshgrid(diameter_values, temperature_values)
    
    diameter = np.exp(D.flatten())
    temperature = T.flatten()
    pressure = np.full_like(diameter, pressure_constant)
    
    v_t = beard_terminal_velocity_numba(
        diameter,
        temperature,
        pressure,
    )
    
    density = pressure / temperature / R_air
    dynamic_viscosity = calc_dynamic_viscosity(temperature)

    return pd.DataFrame({
        'diameter': diameter,
        'temperature': temperature,
        'pressure': pressure,
        'v_t': v_t,
        'density': density,
        'dynamic_viscosity': dynamic_viscosity
    })

def generate_reference_data(size, temperature=285, pressure=90_000):
    diameter = np.logspace(np.log10(1e-6), np.log10(7e-3), size)
    temperature = np.full_like(diameter, temperature)
    pressure = np.full_like(diameter, pressure)
    v_t = beard_terminal_velocity_numba(diameter, temperature, pressure)

    density = pressure / temperature / R_air
    dynamic_viscosity = calc_dynamic_viscosity(temperature)

    df = pd.DataFrame({
        'diameter': diameter,
        'temperature': temperature,
        'pressure': pressure,
        'v_t': v_t,
        'density': density,
        'dynamic_viscosity': dynamic_viscosity
    })

    return df
    
if __name__ == "__main__":
    test_df = generate_sample()
    print(test_df.describe())