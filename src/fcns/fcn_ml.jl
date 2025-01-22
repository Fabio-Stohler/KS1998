@doc raw"""
    normalize_k(kts, mpar)

Function that normalizes the capital stock to represent percentage deviations from the steady state.

# Arguments
- `kts::Array{Float64, 1}`: Capital stock
- `mpar::ModelParameters`: Model parameters

# Returns
- `kts_normalized::Array{Float64, 1}`: Normalized capital stock
"""
function normalize_k(kts, mpar)
    return kts # (kts .- mpar.k_ss) ./ mpar.k_ss # kts
end

@doc raw"""
    create_data(kts, agts, mpar)

Function that creates the data for the neural network.

# Arguments
- `kts::Array{Float64, 1}`: Capital stock
- `agts::Array{Float64, 1}`: Aggregate shock
- `mpar::ModelParameters`: Model parameters

# Returns
- `data::Array{Tuple{Array{Float64, 2}, Array{Float64, 1}}, 1}`: Data for the neural network
"""
function create_data(kts, agts, mpar)
    x = hcat(normalize_k(kts, mpar), agts)[1:(end - 1), :]'
    y = normalize_k(kts[2:end], mpar)
    return [(x, y)]
end

@doc raw"""
    generate_model(mlpar)

Generate a model with the given parameters.

# Arguments
- `mlpar::MLParameters`: Machine learning parameters

# Returns
- `model::Chain`: Neural network model
"""
function generate_model(mpar, mlpar)
    layers = []
    push!(
        layers,
        Dense(mlpar.n_input, mlpar.n_nodes, mlpar.act_in; init = Flux.identity_init),
    )
    for i = 1:(mlpar.n_hidden - 1)
        push!(
            layers,
            Dense(mlpar.n_nodes, mlpar.n_nodes, mlpar.act_mid; init = Flux.identity_init),
        )
    end
    push!(
        layers,
        Dense(mlpar.n_nodes, mlpar.n_output, mlpar.act_out; init = Flux.identity_init),
    )
    return Chain(layers...)
end

@doc raw"""
    predict(model, x, mpar)

Function that predicts the capital stock tomorrow.

# Arguments
- `model`: Neural network model
- `x::Array{Float64, 2}`: Input data
- `mpar::ModelParameters`: Model parameters

# Returns
- `kts_pred::Array{Float64, 1}`: Predicted capital stock
"""
function predict(model, x, mpar)
    return model(x)' #.* mpar.k_ss .+ mpar.k_ss
end

@doc raw"""
    combine_models(model1, model2, p)

Function that combines two neural networks with a given weight.

# Arguments
- `model1`: Neural network model 1
- `model2`: Neural network model 2
- `p::Float64`: Weight

# Returns
- `model::Chain`: Combined neural network model
"""
function combine_models(model1, model2, p)
    # empty list to store the layers
    layers = []

    # combine the two models
    comb_params = p .* Flux.params(model1) .+ (1 .- p) .* Flux.params(model2)
    for i = 1:div(length(comb_params), 2)
        push!(layers, Dense(comb_params[i + (i - 1)], comb_params[2 * i]))
    end
    model = Chain(layers...)
    return model
end
