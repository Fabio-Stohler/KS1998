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
    k::Array{Float64,1} =
        exp.(range(0; stop = log(k_max - k_min + 1.0), length = ngridk)) .+ k_min .- 1.0
    km::Array{Float64,1} = range(km_min, km_max, ngridkm)
    ϵ::Array{Float64,1} = range(0.0, nstates_id - 1.0)
    a::Array{Float64,1} = [1 - δ_a, 1 + δ_a]

    # Meshes for EGM
    mesh_k::Array{Float64} =
        repeat(reshape(k, (ngridk, 1, 1, 1)); outer = [1, ngridkm, nstates_id, nstates_ag])
    mesh_km::Array{Float64} =
        repeat(reshape(km, (1, ngridkm, 1, 1)); outer = [ngridk, 1, nstates_id, nstates_ag])
    mesh_a::Array{Float64} =
        repeat(reshape(a, (1, 1, 1, nstates_ag)); outer = [ngridk, ngridkm, nstates_id, 1])
    mesh_ϵ::Array{Float64} =
        repeat(reshape(ϵ, (1, 1, nstates_id, 1)); outer = [ngridk, ngridkm, 1, nstates_ag])

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
    ag_shock::Array{Int,1} = simulate(MarkovChain(Π_ag), T; init = 1) # start from the bad state

    # Convergence Parameters
    ϵ_k::Float64 = 1e-10
    ϵ_B::Float64 = 1e-8
    update_B::Float64 = 0.3
    iter_max::Int = 100
    iter_max_k::Int = 10000

    # Initial distribution
    distr_init::Array{Float64} = initial_distr(ngridk, nstates_id, k, mpar.k_ss)
end

# Structure containing all grids
@with_kw struct NumericalParametersDelta
    # Model parameters set in advance
    mpar::ModelParameters = ModelParameters()

    # Boundaries for asset grids
    k_min::Int = 0
    k_max::Int = 250
    km_min::Int = 30
    km_max::Int = 60

    # Number of respective gridspoints
    ngridk::Int = 100
    ngridkm::Int = 4
    nstates_id::Int = 2          # number of states for the idiosyncratic shock
    nstates_ag::Int = 2          # number of states for the aggregate shock

    # Parameters for simulation
    burn_in::Int = 100
    T::Int = 1000 + burn_in
    δ_δ::Float64 = 0.005

    # Actual grids
    k::Array{Float64,1} =
        exp.(range(0; stop = log(k_max - k_min + 1.0), length = ngridk)) .+ k_min .- 1.0
    km::Array{Float64,1} = range(km_min, km_max, ngridkm)
    ϵ::Array{Float64,1} = range(0.0, nstates_id - 1.0)
    δ::Array{Float64,1} = [mpar.δ + δ_δ, mpar.δ - δ_δ]

    # Meshes for EGM
    mesh_k::Array{Float64} =
        repeat(reshape(k, (ngridk, 1, 1, 1)); outer = [1, ngridkm, nstates_id, nstates_ag])
    mesh_km::Array{Float64} =
        repeat(reshape(km, (1, ngridkm, 1, 1)); outer = [ngridk, 1, nstates_id, nstates_ag])
    mesh_δ::Array{Float64} =
        repeat(reshape(δ, (1, 1, 1, nstates_ag)); outer = [ngridk, ngridkm, nstates_id, 1])
    mesh_ϵ::Array{Float64} =
        repeat(reshape(ϵ, (1, 1, nstates_id, 1)); outer = [ngridk, ngridkm, 1, nstates_ag])

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
    ag_shock::Array{Int,1} = simulate(MarkovChain(Π_ag), T; init = 1) # start from the bad state

    # Convergence Parameters
    ϵ_k::Float64 = 1e-10
    ϵ_B::Float64 = 1e-8
    update_B::Float64 = 0.3
    iter_max::Int = 100
    iter_max_k::Int = 10000

    # Initial distribution
    distr_init::Array{Float64} = initial_distr(ngridk, nstates_id, k, mpar.k_ss)
end

# Structure containing all grids
@with_kw struct NumericalParametersBeta
    # Model parameters set in advance
    mpar::ModelParameters = ModelParameters()

    # Boundaries for asset grids
    k_min::Int = 0
    k_max::Int = 250
    km_min::Int = 30
    km_max::Int = 60

    # Number of respective gridspoints
    ngridk::Int = 100
    ngridkm::Int = 4
    nstates_id::Int = 2          # number of states for the idiosyncratic shock
    nstates_ag::Int = 2          # number of states for the aggregate shock

    # Parameters for simulation
    burn_in::Int = 100
    T::Int = 1000 + burn_in
    δ_β::Float64 = 0.005

    # Actual grids
    k::Array{Float64,1} =
        exp.(range(0; stop = log(k_max - k_min + 1.0), length = ngridk)) .+ k_min .- 1.0
    km::Array{Float64,1} = range(km_min, km_max, ngridkm)
    ϵ::Array{Float64,1} = range(0.0, nstates_id - 1.0)
    β::Array{Float64,1} = [mpar.β + δ_β, mpar.β - δ_β]

    # Meshes for EGM
    mesh_k::Array{Float64} =
        repeat(reshape(k, (ngridk, 1, 1, 1)); outer = [1, ngridkm, nstates_id, nstates_ag])
    mesh_km::Array{Float64} =
        repeat(reshape(km, (1, ngridkm, 1, 1)); outer = [ngridk, 1, nstates_id, nstates_ag])
    mesh_β::Array{Float64} =
        repeat(reshape(β, (1, 1, 1, nstates_ag)); outer = [ngridk, ngridkm, nstates_id, 1])
    mesh_ϵ::Array{Float64} =
        repeat(reshape(ϵ, (1, 1, nstates_id, 1)); outer = [ngridk, ngridkm, 1, nstates_ag])

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
    ag_shock::Array{Int,1} = simulate(MarkovChain(Π_ag), T; init = 1) # start from the bad state

    # Convergence Parameters
    ϵ_k::Float64 = 1e-10
    ϵ_B::Float64 = 1e-8
    update_B::Float64 = 0.3
    iter_max::Int = 100
    iter_max_k::Int = 10000

    # Initial distribution
    distr_init::Array{Float64} = initial_distr(ngridk, nstates_id, k, mpar.k_ss)
end

@with_kw struct MLParameters
    # Model parameters set in advance
    mpar::ModelParameters = ModelParameters()

    # number of input nodes
    n_input::Int = 2

    # number of hidden layers
    n_hidden::Int = 1

    # number of nodes in each hidden layer
    n_nodes::Int = 32

    # number of output nodes
    n_output::Int = 1

    # number of epochs
    n_epochs::Int = 1000

    # activation functions (first, second, output layer)
    act_in::Function = Flux.sigmoid
    act_mid::Function = Flux.sigmoid
    act_out::Function = Flux.identity

    # loss function
    loss_f::Function = Flux.mse

    # optimizer
    opt::Function = Flux.ADAM

    # learning rate (first, second)
    lr::Float64 = 0.001
    lr2::Float64 = 0.00001
end
