# Work Directories

### data
- dynvis_Rogers_Table.csv
    - Dynamic viscoity data to ensure correct dynamic viscosity calculations
- vT_Best1950.csv
    - Terminal velocities across different pressures and diameters
- vT_Gunn1949
    - Terminal velocities across different diameters

### lib
velocity.py contain terminal velocity functions, and their helper functions.
fast_velocity.py uses numba to speed up these calculations
metrics.py is a helper for plotting
sampler.py is a helper to generate data samples of different kinds, using the latin hypercube method, or sampling across only one axis
sympy_helper.py helps convert to learned sympy expressions to Fortran

### models and outputs
max-corr-5.jls is the final correction model with a max complexity of 183
msre-ref-4.jls is the final reference model with a max complexity of 98

Their corresponding hall of fame files are in `Outputs`

### preprocess

reference_data.py generates reference data - data varying along the diameter axis
correction_data.py takes in a reference model, and calculates the correction needed to accurately predict terminal velocity values from a grid of temperature, pressure and diameter values sampled with the latin hypercube method. Additionally it creates a cartesian mesh with more points to optionally train to the correction model on more data. From my testing when we trained with this large amount of data right away the models training performance did worse.
test_data.py generates 1,000,000 test data points using the latin hypercube method and a fixed seed of 12345.

### src
`get_equations.ipynb` is used to convert a hall of fame file to get the corresponding equations and constants

`cubic_spline.ipynb` and `poly_interp.ipynb` were experiments to see if a standard spline or interpolation could fit the data well, and if so with how many terms.
`Parameter_fit.ipynb` was used to try to further optimize the constant parameters, no additional results as symbolicregression.jl already does this.
`compare_analytical.ipynb` is used to ensure that helper functions match with empirical data.

`load_ref.ipynb` takes in a reference and plots it against beard. Optionally can take in a max complexity. Ensures reference models monotonicity across diameter range.
`load_2stage.ipynb` takes in reference and correction, plots it against beard at standard temperature and pressure. Plots relative error distributions.
`pareto_front.ipynb` loads the hall of fame train files and plots the pareto fronts of the reference and correction models.

`model_to_fortran.ipynb` takes in a model and a ma6x complexity and outputs the corresponding fortran function
`load.f90` contains comparative models such as beard, and the final trained models and measures their execution time and accuracy results
compiled with `gfortran -Ofast -march=native load.f90 -o load`


`plot_error.ipynb` plots 2d heatmap of relative and absolute errors of simmel model, as well as the trained model, and the limited size trained model.
`compare_models.ipynb` creates diagram of final metrics of error and execution timing generated from the fortran file.

### train

`install.jl` ensures that all used julia packages are installed on the operators machines
`train_reference.jl` takes in one parameter `--output-model` that specifies the name of its output directory and one optional parameter `--pretrained-model` which allows to continual training of an existing model so long as you have the (State, Hall of Fame) serialized in a .jls file.
`train_correction.jl` change `const FILE_NAME =` when running this program to your corresponding reference model. Takes the same parameters as train_reference.jl