@doc_raw"""
    normalize_k(kts, mpar)

Function that normalizes the capital stock to represent percentage deviations from the steady state.

# Arguments
- `kts::Array{Float64, 1}`: Capital stock
- `mpar::ModelParameters`: Model parameters

# Returns
- `kts_normalized::Array{Float64, 1}`: Normalized capital stock
"""
function normalize_k(kts, mpar)
    return (kts .- mpar.k_ss) ./ mpar.k_ss
end

@doc_raw"""
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
    y = (kts[2:end] .- mpar.k_ss) ./ mpar.k_ss
    return [(x, y)]
end

@doc_raw"""
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
    return model(x)' .* mpar.k_ss .+ mpar.k_ss
end
