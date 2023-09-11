# structure containing all parameters
@with_kw struct ModelParameters{T}
    β::T = 0.99        # discount factor
    γ::T = 1.0         # utility function parameters
    α::T = 0.36        # share of capital in production function
    δ::T = 0.025       # depreciation rate
    μ::T = 0.15        # unemployment benefits as a share of the wage
    l_bar::T = 1 / 0.9 # time endowment; normalizes s.t. L_bad = 1
    k_ss::T = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
end


# Structure containing all grids
@with_kw struct NumericalParameters
    # Model parameters set in advance
    mpar::ModelParameters = ModelParameters()
    
    # Boundaries for asset grids
    k_min::Int = 0
    k_max::Int = 250
    km_min::Int = 30
    km_max::Int = 50

    # Number of respective gridspoints
    ngridk::Int = 100
    ngridkm::Int = 4
    nstates_id::Int = 2          # number of states for the idiosyncratic shock
    nstates_ag::Int = 2          # number of states for the aggregate shock
    
    # Parameters for simulation
    burn_in::Int = 100
    T::Int = 1000 + burn_in
    δ_a::Float64 = 0.01

    # Actual grids
    x::Array{Float64, 1} = range(0, 0.5, ngridk)
    τ::Int = 3
    y::Array{Float64, 1} = (x ./ maximum(x)) .^ τ
    k::Array{Float64, 1} = k_min .+ (k_max .- k_min) .* y
    km::Array{Float64, 1} = range(km_min, km_max, ngridkm)
    ϵ::Array{Float64, 1} = range(0.0, nstates_id - 1.0)
    a::Array{Float64, 1} = [1 - δ_a, 1 + δ_a]

    # Employment / Unemployment rates
    ur_b::Float64 = shocks_parameters()[1]
    er_b::Float64 = shocks_parameters()[2]
    ur_g::Float64 = shocks_parameters()[3]
    er_g::Float64 = shocks_parameters()[4]
    
    # Transition probabilities
    Π::Matrix{Float64} = shocks_parameters()[5]
    Π_ag::Matrix{Float64} = shocks_parameters()[6]
    
    # Series of aggregate shocks
    seed =  Random.seed!(123)       # Setting a random seed
    ag_shock::Array{Int, 1} = simulate(MarkovChain(Π_ag), T, init = 1) # start from the bad state

    # Convergence Parameters
    dif_B::Float64 = 10^10 # difference between coefficients B of ALM on succ. iter.
    ϵ_k::Float64 = 1e-8
    ϵ_B::Float64 = 1e-6
    update_k::Float64 = 0.77
    update_B::Float64 = 0.3
    iter_max::Int = 100

    # Initial distribution
    distr_init::Array{Float64} = initial_distr(ngridk, nstates_id, k, mpar.k_ss)
end
