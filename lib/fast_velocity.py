import numpy as np
from numba import njit

g = 9.81  # m/s^2
R_air = 287.05  # J / kg / K
R_h2o = 461.52 # J / kg / K

k1 = 1.2e8
k2 = 8000
k3 = 201
@njit()
def cambridge_terminal_velocity_numba(d_m: np.ndarray) -> np.ndarray:
    n = d_m.shape[0]
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        r = d_m[i] / 2
        
        if r < 0.03e-3:
            result[i] = k1 * r ** 2
        elif r < 0.06e-3:
            result[i] = k2 * r
        else:
            result[i] = k3 * r ** 0.5
    
    return result

@njit()
def stokes_velocity_numba(d_m: np.ndarray, density_droplet: np.ndarray, dynamic_viscosity: np.ndarray) -> np.ndarray:
    r = d_m / 2
    return (2 / 9) * (r ** 2) * g * density_droplet / dynamic_viscosity

SIMMEL_ALPHAS = np.array([4.5795e5, 4.962e3, 1.732e3, 9.17e2])
SIMMEL_BETAS = np.array([2.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0, 0.0])
SIMMEL_THRESHOLDS = np.array([0.00013443, 0.00151154, 0.00347784])
RHO_h2o = 1e3 #kg/m^3

@njit()
def simmel_velocity_numba(diameter: np.ndarray, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    n = diameter.shape[0]
    result = np.empty(n, dtype=np.float64)

    for i in range(n):
        d = diameter[i]
        r = d / 2.0
        volume = (4.0 / 3.0) * np.pi * r ** 3
        mass_g = RHO_h2o * volume * 1e3 #convert kg -> g
        
        if d < SIMMEL_THRESHOLDS[0]:
            v_t = SIMMEL_ALPHAS[0] * (mass_g ** SIMMEL_BETAS[0])
        elif d < SIMMEL_THRESHOLDS[1]:
            v_t = SIMMEL_ALPHAS[1] * (mass_g ** SIMMEL_BETAS[1])
        elif d < SIMMEL_THRESHOLDS[2]:
            v_t = SIMMEL_ALPHAS[2] * (mass_g ** SIMMEL_BETAS[2])
        else:
            v_t = SIMMEL_ALPHAS[3] * mass_g
        
        result[i] = 1e-2 * v_t #cm/s to m/s

    return result

@njit()
def simmel_adjusted_velocity_numba(diameter: np.ndarray, temperature: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    n = diameter.shape[0]
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        d = diameter[i]
        r = d / 2.0
        volume = (4.0 / 3.0) * np.pi * r ** 3
        mass_g = RHO_h2o * volume * 1e3 #convert kg -> g
        
        if d < SIMMEL_THRESHOLDS[0]:
            v_t = SIMMEL_ALPHAS[0] * (mass_g ** SIMMEL_BETAS[0])
        elif d < SIMMEL_THRESHOLDS[1]:
            v_t = SIMMEL_ALPHAS[1] * (mass_g ** SIMMEL_BETAS[1])
        elif d < SIMMEL_THRESHOLDS[2]:
            v_t = SIMMEL_ALPHAS[2] * (mass_g ** SIMMEL_BETAS[2])
        else:
            v_t = SIMMEL_ALPHAS[3] * mass_g

        air_density_correction = 1
        
        if pressure[i] <= 40_000 and temperature[i] < 253.15:
            rho_air = pressure[i] / (R_air * temperature[i])
            air_density_correction = (1.2 / rho_air) ** 0.54

        result[i] = 1e-2 * v_t * air_density_correction

    return result

# from Vargaftik+ (1983)
B = 235e-3 # [N/m]
b = -0.625 # []
mu = 1.256 # []
Tc = 647.15 # [K]
@njit
def calc_surface_tension(T: float) -> float:
    sigma =  B*((Tc-T)/Tc)**mu*(1 + b*((Tc-T)/Tc))
    return sigma

@njit
def calc_dynamic_viscosity(T: float) -> float:
    return 1.72e-5*(393. / (T + 120.)) * (T / 273.)**1.5 # [kg m⁻¹ s⁻¹]

beta1 = np.array([
    -0.318657e+1,
    +0.992696,
    -0.153193e-2,
    -0.987059e-3,
    -0.578878e-3,
    +0.855176e-4,
    -0.327815e-5
])

beta2 = np.array([
    -0.500015e+1,
    +0.523778e+1,
    -0.204914e+1,
    +0.475294,
    -0.542819e-1,
    +0.238449e-2
])
density_liquid = 998.0  # kg / m^3
@njit()
def beard_terminal_velocity_numba(diameters: np.ndarray, temperatures: np.ndarray, pressures: np.ndarray) -> np.ndarray:
    n = diameters.shape[0]
    result = np.empty(n, dtype=np.float64)
    
    for i in range(n):
        diameter = diameters[i]
        temperature = temperatures[i]
        pressure = pressures[i]
        
        if temperature <= 0 or diameter <= 0:
            result[i] = np.nan
            continue
            
        density_air = pressure / temperature / R_air  # kg / m^3
        density_difference = density_liquid - density_air
        
        if density_difference <= 0 or density_air <= 0:
            result[i] = np.nan
            continue

        dynamic_viscosity = calc_dynamic_viscosity(temperature)
        surface_tension = calc_surface_tension(temperature)
        
        if dynamic_viscosity <= 0:
            result[i] = np.nan
            continue
                
        if diameter < 19e-6:  # 19 μm in meters
            C_1 = density_difference * g / (18 * dynamic_viscosity)
            result[i] = C_1 * (diameter ** 2)        
        elif diameter < 1.07e-3:  # 1.07 mm in meters
            C_2 = 4 * density_air * density_difference * g / (3 * dynamic_viscosity ** 2)
            N_da = C_2 * (diameter ** 3)
            
            if N_da <= 0:
                result[i] = np.nan
                continue
                
            X = np.log(N_da)

            Y = beta1[0] + X * (beta1[1] + X * (beta1[2] + X * (beta1[3] + X * (beta1[4] + X * (beta1[5] + X * beta1[6])))))
            N_re = np.exp(Y)
            result[i] = dynamic_viscosity * N_re / (density_air * diameter)
        else:
            if surface_tension <= 0:
                result[i] = np.nan
                continue
                
            N_p = surface_tension ** 3 * density_air ** 2 / (dynamic_viscosity ** 4 * density_difference * g)
            C_3 = 4 * density_difference * g / (3 * surface_tension)
            Bo = C_3 * diameter ** 2
            
            if N_p <= 0:
                result[i] = np.nan
                continue
                
            N_p_sixth = N_p ** (1. / 6)
            
            if Bo * N_p_sixth <= 0:
                result[i] = np.nan
                continue
                
            X = np.log(Bo * N_p_sixth)
            
            Y = beta2[0] + X * (beta2[1] + X * (beta2[2] + X * (beta2[3] + X * (beta2[4] + X * beta2[5]))))
                
            N_re = N_p_sixth * np.exp(Y)
            result[i] = dynamic_viscosity * N_re / (density_air * diameter)

    return result