using CSV
using DataFrames
using Random
using Serialization
using Dates
import SymbolicRegression: equation_search, Options, SRRegressor, Dataset, eval_tree_array
using Distributed
using Statistics
using Serialization

if Threads.nthreads() == 1
    @warn "Only 1 thread detected.  Re-run with `julia -t auto` (or `JULIA_NUM_THREADS=<n>`) to use all CPU cores." n_threads = Threads.nthreads()
end

@everywhere using CSV, DataFrames, Random, Serialization, Dates, SymbolicRegression

function parse_output_model()
    i = findfirst(==("--output-model"), ARGS)
    return (i !== nothing && i < length(ARGS)) ? ARGS[i + 1] : nothing
end

function parse_input_model()
    i = findfirst(==("--pretrained-model"), ARGS)
    return (i !== nothing && i < length(ARGS)) ? ARGS[i + 1] : nothing
end

output_model = parse_output_model()
pretrained_model = parse_input_model()

if output_model === nothing
    println("Usage: julia script.jl --output-model <filename>")
    exit(1)
end

const FILE_NAME = "msre-ref-4" #TODO: change to your corresponding reference model
const DATA_PATH = joinpath(@__DIR__, "..", "data", "final_correction_data_" * FILE_NAME * ".csv")
if !isfile(DATA_PATH)
    @error "Could not load" DATA_PATH
    exit(1)
end
df = CSV.read(DATA_PATH, DataFrame)
@info "Data loaded" n_samples = nrow(df)

train_columns = ["diameter", "temperature", "density", "dynamic_viscosity"]
output_column = "correction"

train_syms  = Symbol.(train_columns)
output_sym  = Symbol(output_column)

X = permutedims(Matrix(df[:, Symbol.(train_columns)])) 
y = Vector(df[:, output_sym])

function max_relative_error(tree, dataset::Dataset{T,L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return L(Inf)
    
    eps = T(1e-8)
    rel_errors = abs.((prediction .- dataset.y) ./ (abs.(dataset.y) .+ eps))
    
    return L(maximum(rel_errors))
end

function mean_relative_error(tree, dataset::Dataset{T, L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return L(Inf)
    
    eps = T(1e-8)
    rel_errors = abs.((prediction .- dataset.y) ./ (abs.(dataset.y) .+ eps))
    
    return L(mean(rel_errors))
end

function mean_relative_squared_error(tree, dataset::Dataset{T, L}, options) where {T,L}
    prediction, flag = eval_tree_array(tree, dataset.X, options)
    !flag && return L(Inf)
    
    eps = T(1e-8)
    rel_errors = ((prediction .- dataset.y) ./ (abs.(dataset.y) .+ eps)) .^ 2
    
    return L(mean(rel_errors))
end

# SAFE OPERATORS
@inline safe_pow(x::Real, y::Real) = (x ≥ 0 || isinteger(y)) ? x ^ y : NaN
@inline safe_sqrt(x::Real) = x ≥ 0 ? sqrt(x) : NaN
pow(x::Real, y::Real) = safe_pow(x, y)

# Generate power functions
square(x) = x ^ 2
cube(x) = x ^ 3
pow4(x) = x^4
pow8(x) = x^8
pow12(x) = x^12
pow16(x) = x^16
inv(x) = 1/x

options = Options(
    maxsize=200,
    binary_operators=[+, -, /, *, pow, min, max],
    unary_operators=[abs, inv, safe_sqrt, square, cube, pow4, pow8, pow12, pow16],
    complexity_of_operators=merge(Dict{Function,Float64}(
        min     => 0.5,
        max     => 0.5,
        abs     => 0.5,
        (+)     => 1,
        (-)     => 1,
        (*)     => 1,
        
        square  => 1.0,
        cube    => 2.0,
        pow4    => 2.0,
        pow8    => 3.0,
        pow12   => 4.0,
        pow16   => 4.0,

        inv     => 4.0,
        (/)     => 4.0,
        safe_sqrt   => 11.0,
        pow    => 20,
    )),
    populations=12,
    population_size=150,
    ncycles_per_iteration=12_000,
    parsimony=0.001,
    adaptive_parsimony_scaling=5000.0,
    verbosity=0,
    batching=true,
    batch_size=128,
    annealing=true,
    constraints=Dict(pow => (-1, 5)),
    output_directory="../outputs",
    loss_function=max_relative_error
)

parallel_strategy = :multithreading
@info "Parallelism" strategy = parallel_strategy nthreads = Threads.nthreads()

initial_state = nothing
if pretrained_model != nothing
    @info "Using pretrained model" pretrained_model
    
    pretrained_path = joinpath(@__DIR__, "..", "models", pretrained_model * ".jls")
    if isfile(pretrained_path)
        @info "Loading pretrained state from" pretrained_path
        initial_state = open(pretrained_path, "r") do io
            deserialize(io)
        end        
    else
        @error "Pretrained model file not found" pretrained_path
        exit(1)
    end
end

state, hall_of_fame = equation_search(
    X, y;
    niterations=1_000,
    options=options,
    parallelism=parallel_strategy,
    run_id=output_model,
    return_state=true,
    saved_state = initial_state
)

const MODEL_PATH = joinpath(@__DIR__, "..", "models", output_model * ".jls")
open(MODEL_PATH, "w") do io
    serialize(io, (state, hall_of_fame))
end

println("\nTraining completed!")