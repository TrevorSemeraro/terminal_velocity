from sympy.logic.boolalg import Boolean
from typing import Iterable
import pandas as pd
import sympy as sp
import textwrap
import numpy as np
from scipy.interpolate import LSQUnivariateSpline
import re

_inv, _cube, _square, _safe_sqrt = map(sp.Function, ("inv", "cube", "square", "safe_sqrt"))

_pow4, _pow8, _pow12, _pow16 = map(sp.Function, ("pow4", "pow8", "pow12", "pow16"))

def _safe_sqrt_sym(arg):
    """safe_sqrt(x)  →  sqrt(max(x, 0))  (SymPy version)."""
    return sp.sqrt(sp.Max(arg, 0.0))

def normalise(expr: sp.Expr) -> sp.Expr:
    """Replace inv(), cube(), square(), safe_sqrt(), and pow4-pow16 with pure SymPy."""
    with sp.evaluate(False):
        is_ = lambda f: lambda e: e.func == f
        
        expr = expr.replace(is_(_inv),        lambda e: sp.Pow(e.args[0], -1))
        expr = expr.replace(is_(_cube),       lambda e: sp.Pow(e.args[0], 3))
        expr = expr.replace(is_(_square),     lambda e: sp.Pow(e.args[0], 2))
        expr = expr.replace(is_(_safe_sqrt),  lambda e: _safe_sqrt_sym(e.args[0]))
        
        expr = expr.replace(is_(_pow4),       lambda e: sp.Pow(e.args[0], 4))
        expr = expr.replace(is_(_pow8),       lambda e: sp.Pow(e.args[0], 8))
        expr = expr.replace(is_(_pow12),      lambda e: sp.Pow(e.args[0], 12))
        expr = expr.replace(is_(_pow16),      lambda e: sp.Pow(e.args[0], 16))

    return expr


def generate_fortran_code(file, max_complexity=None):
    csv_path = f"../outputs/{file}/hall_of_fame.csv"
    equation_df = pd.read_csv(csv_path)
    if max_complexity == None:
        best_equation = equation_df.iloc[-1]['Equation']
    else:
        mask = equation_df["Complexity"] <= max_complexity
        idx = equation_df.loc[mask, "Complexity"].idxmax()
        best_equation = equation_df.loc[idx]['Equation']

    sympy_expr = sp.sympify(best_equation)
    sympy_expr_clean = normalise(sympy_expr)
    
    fortran_code = sp.fcode(sympy_expr_clean,
                            assign_to="res",
                            source_format="free",
                            standard=95)
    return sympy_expr_clean, fortran_code

def get_function_from_output(file, max_complexity=None):
    csv_path = f"../outputs/{file}/hall_of_fame.csv"
    equation_df = pd.read_csv(csv_path)
    if max_complexity == None:
        best_equation = equation_df.iloc[-1]['Equation']
    else:
        mask = equation_df["Complexity"] <= max_complexity
        idx = equation_df.loc[mask, "Complexity"].idxmax()
        best_equation = equation_df.loc[idx]['Equation']

    sympy_expr = sp.sympify(best_equation)

    sympy_expr_clean = normalise(sympy_expr)

    vars_      = sorted(sympy_expr_clean.free_symbols, key=lambda s: s.name)
    python_fn  = sp.lambdify(vars_, sympy_expr_clean, modules=['numpy'])

    return python_fn

def wrap_as_function(code, name, args, return_type="real(kind(0d0))", input_types=None, max_complexity=None):
    if input_types is None:
        input_types = {arg: "real(kind(0d0))" for arg in args}
    
    arg_decls = []
    for arg in args:
        arg_type = input_types.get(arg, "real(kind(0d0))")
        arg_decls.append(f"    {arg_type}, INTENT(IN) :: {arg}")
    
    if max_complexity is not None:
        name = f"min_{name}"            
    
    signature = f"PURE ELEMENTAL FUNCTION {name}({', '.join(args)}) result(res)"
    
    func_code = f"""{signature}
        IMPLICIT NONE
        {chr(10).join(arg_decls)}
        {return_type} :: res
        {code}
    END FUNCTION {name}"""
    
    return func_code

def create_spline_sympy_function(data_generator, n_samples=1000, n_knots=8, poly_degree=8, x_col='diameter', y_col='v_t', spline_degree=3):
    df = data_generator(n_samples)
    
    log_x = np.log10(df[x_col])
    log_y = np.log10(df[y_col])
    
    log_x_min, log_x_max = log_x.min(), log_x.max()
    knots = np.linspace(
        log_x_min + 0.1 * (log_x_max - log_x_min),
        log_x_max - 0.1 * (log_x_max - log_x_min),
        n_knots
    )
    
    spline = LSQUnivariateSpline(log_x.values, log_y.values, knots, k=spline_degree)
    
    log_x_eval = np.linspace(log_x.min(), log_x.max(), 1000)
    log_y_eval = spline(log_x_eval)
    
    poly_coeffs = np.polyfit(log_x_eval, log_y_eval, deg=poly_degree)
    
    x_sym = sp.Symbol('x')
    poly_expr = sum(c * x_sym**i for i, c in enumerate(poly_coeffs[::-1]))
    
    input_var = sp.Symbol('d')
    log_input = sp.log(input_var, 10)
    final_expr = 10 ** poly_expr.subs(x_sym, log_input)
    
    func = sp.lambdify(input_var, final_expr, modules='numpy')
    
    spline_info = {
        'knots': knots,
        'degree': spline_degree,
        'poly_degree': poly_degree,
        'data_range_log': (log_x.min(), log_x.max()),
        'data_range_original': (df[x_col].min(), df[x_col].max()),
        'n_samples': n_samples,
        'poly_coeffs': poly_coeffs
    }
    
    return final_expr, func, spline_info