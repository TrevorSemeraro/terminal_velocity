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

const DATA_PATH = joinpath(@__DIR__, "..", "data", "reference_data.csv")
if !isfile(DATA_PATH)
    @error "Could not load" DATA_PATH
    exit(1)
end
df = CSV.read(DATA_PATH, DataFrame)
@info "Data loaded" n_samples = nrow(df)

X = Matrix(df[:, [:diameter]])'
y = Vector(df.v_t)
@assert size(X, 2) == length(y)

# SAFE OPERATORS
@inline safe_pow(x::Real, y::Real) = (x ≥ 0 || isinteger(y)) ? x ^ y : NaN
@inline safe_sqrt(x::Real) = x ≥ 0 ? sqrt(x) : NaN
pow(x::Real, y::Real) = safe_pow(x, y)
square(x) = x ^ 2
cube(x) = x ^ 3
inv(x) = 1/x

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
    base_loss = mean(rel_errors)

    # Diameters dataset are already sorted
    # diffs = diff(prediction)
    # penalty = sum(abs2, min.(diffs, 0))

    return L(mean(rel_errors))
end

pow4(x) = x^4
pow5(x) = x^5

options = Options(
    maxsize=100,
    binary_operators=[+, -, /, *, pow],
    unary_operators=[safe_sqrt, square, cube, inv, abs, pow4, pow5],
    complexity_of_operators=[
        # max     => 0.5,
        # min     => 0.5,
        abs     => 0.5,
        (+)     => 1,
        (-)     => 1,
        (*)     => 1,
        
        square  => 1,
        cube    => 2,
        pow4    => 2,
        pow5    => 3,

        inv     => 4,
        (/)     => 4,
        safe_sqrt   => 11,
        pow    => 20,
    ],
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
    loss_function=mean_relative_squared_error,
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