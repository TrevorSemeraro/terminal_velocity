import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd
from lib.fast_velocity import beard_terminal_velocity_numba
from lib.sampler import generate_grid_samples, generate_sample
import numpy as np

DEFAULT_STYLE = {
    "tick_font_size": 12,
    "axis_title_font_size": 16, 
    "subplot_title_font_size": 18,
    "main_title_font_size": 20
}

def plot_errors(model_names, predict_fns, size=500, pressure_constant=90_000, style=None):
    if style is None:
        style = DEFAULT_STYLE
    
    if not isinstance(model_names, list):
        model_names = [model_names]
    if not isinstance(predict_fns, list):
        predict_fns = [predict_fns]
    
    if len(model_names) != len(predict_fns):
        raise ValueError("Number of model names must match number of predict functions")
    
    test_df = generate_grid_samples(size=size, pressure_constant=pressure_constant)
    test_df['beard_vt'] = beard_terminal_velocity_numba(test_df['diameter'].values, test_df['temperature'].values, test_df['pressure'].values)
    
    all_model_data = []
    global_abs_max = 0
    global_rel_max = 0
    
    for i, (model_name, predict_fn) in enumerate(zip(model_names, predict_fns)):
        model_df = test_df.copy()
        model_df['pred_vt'] = predict_fn(model_df)
        
        model_df['absolute_error'] = np.abs(model_df['pred_vt'] - model_df['v_t'])
        model_df['relative_error'] = (model_df['absolute_error'] / np.abs(model_df['v_t'])) * 100
        
        model_df['diameter'] = model_df['diameter'] * 1000
        
        global_abs_max = max(global_abs_max, model_df['absolute_error'].max())
        global_rel_max = max(global_rel_max, model_df['relative_error'].max())
        
        all_model_data.append((model_name, model_df))
    
    num_models = len(all_model_data)
    
    abs_fig, abs_axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    if num_models == 1:
        abs_axes = [abs_axes]
    
    rel_fig, rel_axes = plt.subplots(1, num_models, figsize=(6 * num_models, 6))
    if num_models == 1:
        rel_axes = [rel_axes]
    
    for i, (model_name, model_df) in enumerate(all_model_data):
        correlation_pivot_data = {
            "Absolute Error": model_df.pivot_table(index='temperature', columns='diameter', values='absolute_error'),
            "Relative Error": model_df.pivot_table(index='temperature', columns='diameter', values='relative_error'),
        }

        x_vals = correlation_pivot_data["Absolute Error"].columns.values
        y_vals = correlation_pivot_data["Relative Error"].index.values

        tick_vals = np.linspace(x_vals.min(), x_vals.max(), num=5)
        tick_labels = []
        for val in tick_vals:
            exp = int(np.floor(np.log10(val)))
            mant = val / 10**exp
            if exp == 0:
                tick_labels.append(f"{mant:.1f}")
            else:
                tick_labels.append(f"{mant:.1f}×10$^{{{exp}}}$")

        abs_ax = abs_axes[i]
        abs_im = abs_ax.imshow(
            correlation_pivot_data["Absolute Error"].values,
            cmap="Reds",
            aspect='auto',
            origin='lower',
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            vmin=0,
            vmax=global_abs_max
        )
        
        abs_ax.set_title(model_name, fontsize=style["subplot_title_font_size"], pad=8)
        abs_ax.set_xlabel("Diameter (mm)", fontsize=style["axis_title_font_size"])
        abs_ax.tick_params(axis='x', labelsize=style["tick_font_size"])
        abs_ax.tick_params(axis='y', labelsize=style["tick_font_size"])
        abs_ax.set_xticks(tick_vals)
        abs_ax.set_xticklabels(tick_labels)
        
        if i == 0:
            abs_ax.set_ylabel("Temperature (K)", fontsize=style["axis_title_font_size"])
        
        divider = make_axes_locatable(abs_ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(abs_im, cax=cax)
        cbar.set_label("Absolute Error (m/s)", fontsize=style["axis_title_font_size"])
        cbar.ax.tick_params(labelsize=style["tick_font_size"])

        rel_ax = rel_axes[i]
        rel_im = rel_ax.imshow(
            correlation_pivot_data["Relative Error"].values,
            cmap="Oranges",
            aspect='auto',
            origin='lower',
            extent=[x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()],
            vmin=0,
            vmax=global_rel_max
        )
        
        rel_ax.set_title(model_name, fontsize=style["subplot_title_font_size"], pad=8)
        rel_ax.set_xlabel("Diameter (mm)", fontsize=style["axis_title_font_size"])
        rel_ax.tick_params(axis='x', labelsize=style["tick_font_size"])
        rel_ax.tick_params(axis='y', labelsize=style["tick_font_size"])
        rel_ax.set_xticks(tick_vals)
        rel_ax.set_xticklabels(tick_labels)
        
        if i == 0:
            rel_ax.set_ylabel("Temperature (K)", fontsize=style["axis_title_font_size"])
        
        divider = make_axes_locatable(rel_ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(rel_im, cax=cax)
        cbar.set_label("Relative Error (%)", fontsize=style["axis_title_font_size"])
        cbar.ax.tick_params(labelsize=style["tick_font_size"])

    abs_fig.suptitle('Absolute Error Comparison', fontsize=style["main_title_font_size"], y=0.98)
    rel_fig.suptitle('Relative Error Comparison', fontsize=style["main_title_font_size"], y=0.98)
    
    abs_fig.tight_layout()
    rel_fig.tight_layout()
    
    return abs_fig, rel_fig

def error_metrics(predict_fn):
    test_data = generate_sample(num_samples=1_000_000, seed=7044)
    test_data['pred_vt'] = predict_fn(test_data)

    test_data['absolute_error'] = np.abs(test_data['pred_vt'] - test_data['v_t'])
    test_data['relative_error'] = test_data['absolute_error'] / np.abs(test_data['v_t'])

    print(np.mean(test_data['absolute_error']), np.max(test_data['absolute_error']))
    print(np.mean(test_data['relative_error']), np.max(test_data['relative_error']))

    test_data.head(5)