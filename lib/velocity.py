import numpy as np
import pandas as pd
from typing import Union, Dict
import matplotlib.pyplot as plt

# EARTH CONSTANTS
g = 9.81 # m/s^2
R_air = 287.05 # J / kg / K
R_h2o = 461.52 # J / kg / K

def calc_surface_tension(T):
    # from Vargaftik+ (1983)
    B = 235e-3 # [N/m]
    b = -0.625 # []
    mu = 1.256 # []
    Tc = 647.15 # [K]
    sigma =  B*((Tc-T)/Tc)**mu*(1 + b*((Tc-T)/Tc))
    return sigma

def calc_dynamic_viscosity(T):
    return 1.72e-5*(393. / (T + 120.)) * (T / 273.)**1.5 # [kg m⁻¹ s⁻¹]

def beard_terminal_velocity(diameter, temp_air, pressure_air, cunningham_correction=False):
    """
    Derived from Beard
    Parameters
    --------
    Diameter
    - units: m
    temp_air
    - units: Kelvin
    pressure_air:
    - units: Pa
    Returns
    --------
    terminal_velocity : float
        Terminal velocity in m/s
    """   
    if diameter < 5e-7 or diameter > 7.5e-3:  # 0.5 μm to 7 mm in meters
        raise ValueError(f"Diameter {diameter }out of valid range (0.5 μm to 7 mm)")

    density_air = pressure_air / temp_air / R_air  # kg / m^3

    density_liquid = 998.0  # kg / m^3
    
    density_difference = density_liquid - density_air
    
    if density_difference <= 0:
        return np.nan

    # dynamic viscosity of air for modern Earth conditions
    dynamic_viscosity = calc_dynamic_viscosity(temp_air)
    
    # calculate condensible liquid surface tension as a function of temperature
    surface_tension = calc_surface_tension(temp_air)

    # PAPER CONSTANTS
    initial_air_molecules = 6.62e-8  # m
    initial_pressure = 101325  # Pa
    initial_dynamic_viscosity = 1.818e-5  # Pa·s
    initial_temperature = 293.15  # K
    
    C_sc = 1
    if cunningham_correction:    
        air_molecules = initial_air_molecules * (dynamic_viscosity/initial_dynamic_viscosity) * (initial_pressure/pressure_air) * np.sqrt(temp_air/initial_temperature)
        C_sc = 1 + 2.51 * air_molecules / diameter
        
    if diameter < 19e-6:  # 19 μm in meters
        C_1 = density_difference * g / 18 / dynamic_viscosity
        return C_1 * C_sc * (diameter ** 2)
    elif diameter < 1.07e-3:  # 1.07 mm in meters
        C_2 = 4 * density_air * density_difference * g / 3 / (dynamic_viscosity ** 2)
        N_da = C_2 * (diameter ** 3)
        X = np.log(N_da)
        beta_coefficients = np.array([
            -0.318657e+1,
            +0.992696,
            -0.153193e-2,
            -0.987059e-3,
            -0.578878e-3,
            +0.855176e-4,
            -0.327815e-5
        ])
        Y = 0
        for index, coefficient in enumerate(beta_coefficients):
            Y += coefficient * (X ** index)
        N_re = C_sc * np.exp(Y)
        return dynamic_viscosity * N_re / (density_air * diameter)
    else:
        N_p = surface_tension ** 3 * density_air ** 2 / dynamic_viscosity ** 4 / density_difference / g
        C_3 = (4 / 3) * density_difference * g / surface_tension
        Bo = C_3 * diameter ** 2
        X = np.log(Bo * N_p ** (1. /6))
        
        beta_coefficients = np.array([
            -0.500015e+1,
            +0.523778e+1,
            -0.204914e+1,
            +0.475294,
            -0.542819e-1,
            +0.238449e-2
        ])
        Y = 0
        for index, coefficient in enumerate(beta_coefficients):
            Y += coefficient * (X ** index)
            
        N_re = (N_p ** (1. /6)) * np.exp(Y)
        return dynamic_viscosity * N_re / (density_air * diameter)

def cambridge_terminal_velocity(d_m: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    # 7.2.3 Terminal Velocity of cloud droplets and raindrops
    k1 = 1.2e8 # 1/m/s
    k2 = 8000 # 1/s
    k3 = 201 # m^(1/2)/s
    
    r = d_m / 2
    
    # Initialize output array with same shape as input
    if isinstance(d_m, (np.ndarray, pd.Series)):
        v_t = np.zeros_like(d_m, dtype=float)
        mask1 = r < 0.03e-3
        mask2 = (r >= 0.03e-3) & (r < 0.06e-3)
        # TODO: this covers only up to 2mm, at which point raindrops are no longer spherical
        mask3 = r >= 0.06e-3
        
        v_t[mask1] = k1 * r[mask1] ** 2
        v_t[mask2] = k2 * r[mask2]
        v_t[mask3] = k3 * r[mask3] ** (1/2)
        return v_t
    else:
        # Handle scalar input
        if r < 0.03e-3:
            return k1 * r ** 2
        elif r < 0.06e-3:
            return k2 * r
        else:
            return k3 * r ** (1/2)

def stokes_velocity(d_m: Union[float, np.ndarray, pd.Series], 
                   density_droplet: Union[float, np.ndarray, pd.Series], 
                   dynamic_viscosity: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    r_m = d_m / 2
    return (2 / 9) * (r_m ** 2) * g * density_droplet / dynamic_viscosity

R_air = 287.05  # J / kg / K

def simmel_velocity(diameter, temperature, pressure):
    """
    Compute terminal velocity using Simmel et al.'s formulation,
    using droplet mass (in grams) as input to the power law,
    but thresholds still based on diameter (µm).
    
    Parameters:
    - diameter: float or np.array, diameter in meters
    
    Returns:
    - Terminal velocity in m/s
    """
    # rho_air = pressure / temperature / R_air
    # rho_ref = 1.2 # kg / m^3
    # air_density_correction = (rho_air / rho_ref) ** 0.54
    
    d_um = diameter * 1e6  # micrometers
    radius_m = diameter / 2
    volume_m3 = (4/3) * np.pi * radius_m**3
    mass_g = volume_m3 * 1e6  # 1e6 g/m^3 = 1 g/cm^3 × 1e6 cm^3/m^3

    # Breakpoints in micrometers (as in Table 2)
    mask1 = d_um < 134.43
    mask2 = (d_um >= 134.43) & (d_um < 1511.54)
    mask3 = (d_um >= 1511.54) & (d_um < 3477.84)
    mask4 = d_um >= 3477.84

    # Coefficients from Table 2
    alphas = np.array([4.5795e5, 4.962e3, 1.732e3, 9.17e2])  # cm/s
    betas = np.array([2./3., 1./3., 1./6., 0])

    # Initialize velocity array
    v_t = np.zeros_like(mass_g, dtype=float)

    # Apply formula using mass in grams
    v_t[mask1] = alphas[0] * (mass_g[mask1] ** betas[0])
    v_t[mask2] = alphas[1] * (mass_g[mask2] ** betas[1])
    v_t[mask3] = alphas[2] * (mass_g[mask3] ** betas[2])
    v_t[mask4] = alphas[3] * (mass_g[mask4] ** betas[3])

    # return air_density_correction * 1e-2 * v_t  # convert from cm/s to m/s
    return 1e-2 * v_t  # convert from cm/s to m/s

if __name__ == "__main__":
    # test the beard_terminal_velocity function
    diameters_regime1 = np.linspace(1e-6, 20e-6, 50)  # 1 μm to 20 μm (measured in m)
    diameters_regime2 = np.linspace(20e-6, 1e-3, 50)  # 20 μm to 1 mm (measured in m)
    diameters_regime3 = np.linspace(1e-3, 7e-3, 50)  # 1 mm to 7 mm (measured in m)

    temp = 300  # 273-323 Kelvin
    pressure = 101325  # Pa

    terminal_velocity_1 = []
    terminal_velocity_2 = []
    terminal_velocity_3 = []

    for i in range(50):
        terminal_velocity_1.append(beard_terminal_velocity(diameters_regime1[i], temp, pressure))
        terminal_velocity_2.append(beard_terminal_velocity(diameters_regime2[i], temp, pressure))
        terminal_velocity_3.append(beard_terminal_velocity(diameters_regime3[i], temp, pressure))

    plt.scatter(diameters_regime1, terminal_velocity_1)
    plt.scatter(diameters_regime2, terminal_velocity_2)
    plt.scatter(diameters_regime3, terminal_velocity_3)
    plt.xlabel("Diameter, m")
    plt.ylabel("Terminal Velocity, m/s")
    plt.show()
    
    # generate data
    rng = np.random.default_rng(seed=42)
    num_samples = 1_000

    # DIAMETER SIZE RANGES OF CHAPTER 7
    small = rng.uniform(1e-6, 3e-4, size=num_samples)  # 1μm to 0.3mm
    medium = rng.uniform(3e-4, 6e-4, size=num_samples)  # 0.3mm to 0.6mm
    large = rng.uniform(6e-4, 2e-3, size=num_samples)  # 0.6mm to 2mm

    zones = np.random.randint(1, 7, size=num_samples)

    # pressure and temperature ranges of troposphere
    pressures = np.random.uniform(1e4, 1e5, size=num_samples)  # Pa
    temperatures = np.random.uniform(223.15, 298, size=num_samples)

    X = []
    Y = []
    for diameter_idx, diameter_size in enumerate([small, medium, large]):
        
        for i in range(num_samples):
            # X.append([diameter_size[i], temperatures[i], pressures[i]])
            X.append([diameter_size[i]])
            Y.append([beard_terminal_velocity(diameter_size[i], temperatures[i], pressures[i])])

        print(diameter_idx, np.min(diameter_size), np.max(diameter_size))
    np.save(f"data_{diameter_idx}.npy", X)
    np.save(f"labels{diameter_idx}.npy", Y)

    plt.scatter(X, Y)
    plt.xlabel("Diameter, m")
    plt.ylabel("Terminal Velocity, m/s")