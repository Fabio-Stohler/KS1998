# structure containing all parameters
@with_kw struct ModelParameters{T}
    β::T = 0.99        # discount factor
    σ::T = 1.0         # utility function parameters
    α::T = 0.36        # share of capital in production function
    δ::T = 0.025       # depreciation rate
    μ::T = 0.15        # unemployment benefits as a share of the wage
    l_bar::T = 1 / 0.9 # time endowment; normalizes s.t. L_bad = 1
    Rbar::T = 0.0      # interest rate gap from borrowing
    k_ss::T = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
end


# Structure containing all grids
@with_kw struct NumericalParameters
    # Model parameters set in advance
    mpar::ModelParameters = ModelParameters()

    # Boundaries for asset grids
    k_min::Int = 0
    k_max::Int = 250
    b_min::Int = -1.0
    b_max::Int = 50.0
    ω_min::Int = 0.0
    ω_max::Int = 1.0
    x_min::Int = 0.0
    x_max::Int = 100.0
    km_min::Int = 30
    km_max::Int = 50

    # Number of respective gridspoints
    ngridk::Int = 100
    ngridb::Int = 100
    ngridx::Int = 100
    ngridω::Int = 20
    ngridkm::Int = 4
    nstates_id::Int = 2          # number of states for the idiosyncratic shock
    nstates_ag::Int = 2          # number of states for the aggregate shock

    # Parameters for simulation
    burn_in::Int = 100
    T::Int = 1000 + burn_in
    δ_a::Float64 = 0.01

    # Actual grids
    k::Array{Float64,1} =
        exp.(range(0, stop = log(k_max - k_min + 1.0), length = ngridk)) .+ k_min .- 1.0
        km::Array{Float64,1} = range(km_min, km_max, ngridkm)
        ϵ::Array{Float64,1} = range(0.0, nstates_id - 1.0)
        a::Array{Float64,1} = [1 - δ_a, 1 + δ_a]
    b::Array{Float64,1} = exp.(range(0, stop = log(b_max - b_min + 1.0), length = ngridb)) .+ b_min .- 1.0
    x::Array{Float64,1} = exp.(range(0, stop = log(x_max - x_min + 1.0), length = ngridx)) .+ x_min .- 1.0 
    ω::Array{Float64,1} = exp.(range(0, stop = log(ω_max - ω_min + 1.0), length = ngridω)) .+ ω_min .- 1.0

    # Meshes for EGM
    mesh_k::Array{Float64} =
        repeat(reshape(k, (ngridk, 1, 1, 1)), outer = [1, ngridkm, nstates_ag, nstates_id])
    mesh_km::Array{Float64} =
        repeat(reshape(km, (1, ngridkm, 1, 1)), outer = [ngridk, 1, nstates_ag, nstates_id])
    mesh_a::Array{Float64} =
        repeat(reshape(a, (1, 1, nstates_ag, 1)), outer = [ngridk, ngridkm, 1, nstates_id])
    mesh_ϵ::Array{Float64} =
        repeat(reshape(ϵ, (1, 1, 1, nstates_id)), outer = [ngridk, ngridkm, nstates_ag, 1])
    mesh_x::Array{Float64, 4} =
        repeat(reshape(x, (ngridx, 1, 1, 1)), outer = [1, ngridkm, nstates_ag, nstates_id])
    mesh_ω::Array{Float64, 5} = 
        repeat(reshape(ω, (ngridω, 1, 1, 1, 1)), outer = [1, ngridx, ngridkm, nstates_ag, nstates_id])
    mesh_b::Array{Float64, 5} =
        repeat(reshape(b, (1, ngridb, 1, 1, 1)), outer = [ngridω, 1, ngridkm, nstates_ag, nstates_id])

    # Employment / Unemployment rates
    ur_b::Float64 = shocks_parameters()[1]
    er_b::Float64 = shocks_parameters()[2]
    ur_g::Float64 = shocks_parameters()[3]
    er_g::Float64 = shocks_parameters()[4]

    # Transition probabilities
    Π::Matrix{Float64} = shocks_parameters()[5]
    Π_ag::Matrix{Float64} = shocks_parameters()[6]

    # Series of aggregate shocks
    seed = Random.seed!(123)       # Setting a random seed
    ag_shock::Array{Int,1} = simulate(MarkovChain(Π_ag), T, init = 1) # start from the bad state

    # Convergence Parameters
    ϵ_k::Float64 = 1e-10
    ϵ_B::Float64 = 1e-6
    update_B::Float64 = 0.3
    iter_max::Int = 100
    iter_max_k::Int = 10000

    # Initial distribution
    distr_init::Array{Float64} = initial_distr(ngridk, nstates_id, k, mpar.k_ss)
end
