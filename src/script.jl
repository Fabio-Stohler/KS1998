"""
Solution to Krusell and Smith (1998) from Fabio Stohler 
"""

# setting the correct working directory
cd("./src")

# importing all necessary libraries
using QuantEcon, Plots
using Random, Distributions, Dierckx
using Interpolations, GLM
using LinearAlgebra, Statistics
using NLsolve

include("./fcns/LocateFcn.jl")
include("./fcns/fcn_makeweights.jl")

# function to set all parameters
function gen_params()
    β = 0.99        # discount factor
    γ = 1.0         # utility function parameters
    α = 0.36        # share of capital in production function
    δ = 0.025       # depreciation rate
    μ = 0.15        # unemployment benefits as a share of the wage
    l_bar = 1 / 0.9 # time endowment; normalizes s.t. L_bad = 1
    k_ss = (α * β / (1 - β * (1 - δ)))^(1 / (1 - α))
    return α, β, γ, δ, μ, l_bar, k_ss
end

function gen_grid()
    N = 10000       # number of agents for stochastic simulation
    J = 1000        # number of grid points for stochastic simulation
    k_min = 0
    k_max = 250
    burn_in = 100
    T = 1000 + burn_in
    ngridk = 100
    x = range(0, 0.5, ngridk)
    τ = 2
    y = (x ./ maximum(x)) .^ τ
    km_min = 30
    km_max = 50
    k = k_min .+ (k_max .- k_min) .* y
    ngridkm = 4
    km = range(km_min, km_max, ngridkm)
    return (N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm)
end

function shocks_parameters()
    nstates_id = 2          # number of states for the idiosyncratic shock
    nstates_ag = 2          # number of states for the aggregate shock
    ϵ = range(0.0, nstates_id - 1.0)
    δ_a = 0.01
    a = [1 - δ_a, 1 + δ_a]

    # Assumptions on the risk behavior
    D_g = 8                 # Duration of a good aggregate state
    D_b = 8                 # Duration of a bad aggregate state
    ur_b = 0.1              # unemployment rate in a bad aggregate state
    er_b = (1 - ur_b)       # employment rate in a bad aggregate state
    ur_g = 0.04             # unemployment rate in a good aggregate state
    er_g = (1 - ur_g)       # employment rate in a good aggregate state

    # Time to find a job in the respective states
    # Coding is D_ss'ee' where e is the employment state and s is the aggregate state
    D_bb01 = 2.5
    D_gg01 = 1.5

    # Producing the actual transition matrix
    π_gg = 1 - 1 / D_g
    π_bb = 1 - 1 / D_b
    Π_ag = [π_bb 1-π_bb; 1-π_gg π_gg]

    # Transition conditional on the current state
    π_01_bb = 1 / D_bb01
    π_01_gg = 1 / D_gg01

    # Setting up the transition matrix conditional on the states
    Π = zeros((4, 4))
    Π[1, 2] = π_01_bb * π_bb     # Prob(ϵ' = 1, z' = 0| ϵ = 0, z = 0) = Prob(ϵ' = 1| ϵ = 0, z' = 0, z = 0) * Prob(z' = 0| z = 0) 
    Π[1, 1] = π_bb - Π[1, 2]      # Π_bb00 + Π_bb01 = Π_bb
    Π[3, 4] = π_01_gg * π_gg     # Prob(ϵ' = 1, z' = 1| ϵ = 0, z = 1) = Prob(ϵ' = 1| ϵ = 0, z' = 1, z = 1) * Prob(z' = 1| z = 1)
    Π[3, 3] = π_gg - Π[3, 4]      # Π_gg00 + Π_gg01 = Π_gg

    # Conditions for transtion Π_gb00 / Π_gb = 1.25 Π_bb00 / Π_bb
    Π[3, 1] = 1.25 * Π[1, 1] / Π_ag[1, 1] * Π_ag[2, 1]
    Π[3, 2] = Π_ag[2, 1] - Π[3, 1] # Π_gb00 + Π_gb01 = Π_gb

    # Conditions for transition Π_bg00 / Π_bg = 0.75 Π_gg00 / Π_gg
    Π[1, 3] = 0.75 * Π[3, 3] / Π_ag[2, 2] * Π_ag[1, 2]
    Π[1, 4] = Π_ag[1, 2] - Π[1, 3] # Π_bg00 + Π_bg01 = Π_bg

    # Imposing the law of motion for unemployment
    # u_s * Π_ss'00 / Π_ss' + (1 - u_s) * Π_ss'10 / Π_ss' = u_s'
    Π[2, 1] = ur_b * π_bb / (1 - ur_b) * (1 - Π[1, 1] / π_bb)
    Π[2, 2] = π_bb - Π[2, 1] # Π_bb10 + Π_bb11 = Π_bb
    Π[2, 3] = Π_ag[1, 2] / (1 - ur_b) * (ur_g - ur_b * Π[1, 3] / Π_ag[1, 2])
    Π[2, 4] = Π_ag[1, 2] - Π[2, 3] # Π_bg10 + Π_bg11 = Π_bg 
    Π[4, 1] = Π_ag[2, 1] / (1 - ur_g) * (ur_b - ur_g * Π[3, 1] / Π_ag[2, 1])
    Π[4, 2] = Π_ag[2, 1] - Π[4, 1] # Π_gb10 + Π_gb11 = Π_gb
    Π[4, 3] = ur_g * π_gg / (1 - ur_g) * (1 - Π[3, 3] / π_gg)
    Π[4, 4] = π_gg - Π[4, 3] # Π_gg10 + Π_gg11 = Π_gg

    return nstates_id, nstates_ag, ϵ, ur_b, er_b, ur_g, er_g, a, Π
end


function shocks()
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
    nstates_id, nstates_ag, ϵ, ur_b, er_b, ur_g, er_g, a, Π = shocks_parameters()
    ag_shock = zeros(T, 1)
    id_shock = zeros(T, N)
    Random.seed!(123)

    # Transition probabilities between aggregate states
    prob_ag = zeros((nstates_ag, nstates_ag))
    prob_ag[1, 1] = Π[1, 1] + Π[1, 2]
    prob_ag[2, 1] = 1 - prob_ag[1, 1] # bad state to good state
    prob_ag[2, 2] = Π[3, 3] + Π[3, 4]
    prob_ag[1, 2] = 1 - prob_ag[2, 2] # good state to bad state
    P = Π ./ kron(prob_ag, ones(nstates_ag, nstates_ag))

    # Generate aggregate shocks
    mc = MarkovChain(prob_ag)
    ag_shock = simulate(mc, T, init = 1) # start from the bad state

    # generate idiosyncratic shocks for all agents in the first period
    draw = rand(N)
    id_shock[1, :] = (draw .> ur_b) .+ 1.0  # set state to good if probability exceeds ur_b

    # generate idiosyncratic shocks for all agents starting in second period
    draw = Random.rand(T - 1, N)
    for t in range(2, length = T - 1)
        # Fix idiosyncratic transition matrix conditional on aggregate state
        transition = P[ag_shock[t-1]:2*ag_shock[t-1], ag_shock[t]:2*ag_shock[t]]
        transition_prob = [
            transition[Int(id_shock[t-1, i]), Int(id_shock[t-1, i])] for
            i in range(1, length = N)
        ]
        # Check this again!
        check = transition_prob > draw[t-1, :] # sign whether to remain in current state
        id_shock[t, :] =
            (id_shock[t-1, :]) .* check .+ (3 .- id_shock[t-1, :]) .* (1 .- check)
    end
    return id_shock, ag_shock
end



function convergence_parameters(nstates_ag = 2)
    dif_B = 10^10 # difference between coefficients B of ALM on succ. iter.
    ϵ_k = 1e-8
    ϵ_B = 1e-6
    update_k = 0.77
    update_B = 0.3
    B = zeros(nstates_ag, nstates_ag)
    B[:, 1] .= 0.0
    B[:, 2] .= 1.0
    return B, dif_B, ϵ_k, ϵ_B, update_k, update_B
end



# Solving the individual problem
function iterate_policy(
    k_prime::Array,
    K_prime::Array,
    wealth::Array,
    irate::Array,
    wage::Array,
    tax::Array,
    P::Array,
)
    # Extracting necessary stuff
    dif_B, criter_k, criter_B, update_k, update_B = convergence_parameters()[2:end]
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
    alpha, beta, gamma, delta, mu, l_bar, k_ss = gen_params()
    nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()
    replacement = Array([mu, l_bar]) #replacement rate of wage
    n = ngridk * ngridkm * nstates_ag * nstates_id

    # Convergence parameters
    dif_k = 1
    iter_k = 1
    iter_k_max = 20000
    while dif_k > criter_k && iter_k < iter_k_max
        """
            interpolate policy function k'=k(k, km) in new points (k', km')
        """
        k2_prime = zeros((n, nstates_ag, nstates_id))
        c_prime = zeros((n, nstates_ag, nstates_id))
        mu_prime = zeros((n, nstates_ag, nstates_id))

        #reshape k_prime for interpolation
        k_prime_reshape = reshape(k_prime, (ngridk, ngridkm, nstates_id, nstates_ag))
        K_prime_reshape = reshape(K_prime, (ngridk, ngridkm, nstates_id, nstates_ag))
        for i in range(1, nstates_ag)
            for j in range(1, nstates_id)
                # capital in aggregate state i, idiosyncratic state j as a function of current states
                k2_prime[:, i, j] = reshape(
                    evaluate(
                        Spline2D(k, km, k_prime_reshape[:, :, i, j]),
                        k_prime_reshape[:],
                        K_prime_reshape[:],
                    ),
                    n,
                )

                c_prime[:, i, j] = (
                    irate[:, i] .* k_prime .+ replacement[j] .* (wage[:, i]) .+
                    (1 .- delta) .* k_prime .- k2_prime[:, i, j] .- tax[:, i, j]
                )
            end
        end

        # replace negative consumption by very low positive number
        c_prime = max.(c_prime, 10^(-10))
        mu_prime = c_prime .^ (-gamma)

        #Expectation term in Euler equation
        #Components in terms of all possible transitions
        expec_comp = zeros((n, nstates_ag, nstates_id))
        for i in range(1, length = nstates_ag)
            for j in range(1, length = nstates_id)
                expec_comp[:, i, j] =
                    (mu_prime[:, i, j] .* (1 .- delta .+ irate[:, i])) .* P[:, 2*(i-1)+j]
            end
        end
        """
        Expectation term in the Euler equation
        sum over various transitions (which have already been scaled by their probability)
        """
        expec = sum(
            expec_comp[:, i, j] for i in range(1, length = nstates_ag) for
            j in range(1, length = nstates_id)
        )

        # current consumption from Euler equation if borrowing constraint is not binding
        cn = (beta .* expec) .^ (-1 / gamma)
        k_prime_n = wealth - cn
        k_prime_n = min.(k_prime_n, k_max)
        k_prime_n = max.(k_prime_n, k_min)
        """
        difference between new and previous capital functions
        """
        dif_k = norm(k_prime_n - k_prime)
        k_prime = update_k .* k_prime_n .+ (1 .- update_k) .* k_prime  # update k_prime_n
        iter_k += 1
    end
    if iter_k == iter_k_max
        println("EGM did not converge with error: ", dif_k)
    end
    c = wealth - k_prime
    return k_prime, c
end



# Solve the individual problem
function individual(k_prime, B)
    dif_B, criter_k, criter_B, update_k, update_B = convergence_parameters()[2:end]
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
    alpha, beta, gamma, delta, mu, l_bar, k_ss = gen_params()
    nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()
    e = Array([er_b, er_g])
    u = 1 .- e
    #Tax rate depending on aggregate and idiosyncratic states

    #Transition probabilities by current state (k,km, Z, eps) and future (Z', eps')
    n = ngridk * ngridkm * nstates_ag * nstates_id
    P = zeros((ngridk, ngridkm, nstates_ag, nstates_id, nstates_ag * nstates_id))
    for z in range(1, length = nstates_ag * nstates_id)
        for i in range(1, length = nstates_ag)
            for j in range(1, length = nstates_id)
                # Check this again!
                P[:, :, i, j, z] = prob[2*(i-1)+j, z] * ones((ngridk, ngridkm))
            end
        end
    end
    P = reshape(P, (n, nstates_ag * nstates_id))

    k_indices = zeros(n)
    km_indices = zeros(n)
    ag = zeros(n)
    e_i = zeros(n)
    Cart_Indices = CartesianIndices(
        Array(reshape(range(1, length = n), (ngridk, ngridkm, nstates_ag, nstates_id))),
    )
    for s_i in range(1, length = n)
        k_indices[s_i], km_indices[s_i], ag[s_i], e_i[s_i] = Tuple(Cart_Indices[s_i])
    end
    k_indices = Array([Int(x) for x in k_indices])
    km_indices = Array([Int(x) for x in km_indices])
    ag = Array([Int(x) for x in ag])
    e_i = Array([Int(x .- 1.0) for x in e_i])

    """
    Using indices, generate arrays for productivity, employment, aggregate
    capital, individual capital
    """

    Z = Array([a[Int(i)] for i in ag])
    L = Array([e[Int(i)] for i in ag])
    K = Array([km[Int(i)] for i in km_indices])
    k_i = Array([k[Int(i)] for i in k_indices])
    irate = alpha .* Z .* (K ./ (l_bar .* L)) .^ (alpha - 1)
    wage = (1 - alpha) .* Z .* (K ./ (l_bar .* L)) .^ alpha
    wealth =
        irate .* k_i .+ (wage .* e_i) .* l_bar .+ mu .* (wage .* (1 .- e_i)) .+
        (1 .- delta) .* k_i .- mu .* (wage .* (1 .- L) ./ L) .* e_i

    # Transition of capital depends on aggregate state
    K_prime =
        Array([exp(B[ag[i], 1] .+ B[ag[i], 2] * log(K[i])) for i in range(1, length = n)])

    # restrict km_prime to fall into bounds
    K_prime = min.(K_prime, km_max)
    K_prime = max.(K_prime, km_min)

    # Future interst rate and wage conditional on state (bad or good state)
    irate_prime = zeros((n, nstates_ag))
    wage_prime = zeros((n, nstates_ag))
    for i in range(1, nstates_ag)
        irate_prime[:, i] = alpha .* a[i] .* ((K_prime ./ (e[i] .* l_bar)) .^ (alpha .- 1))
        wage_prime[:, i] = (1 .- alpha) .* a[i] .* ((K_prime / (e[i] * l_bar)) .^ alpha)
    end

    # Tax rate
    tax_prime = zeros((n, nstates_ag, nstates_id))
    for i in range(1, nstates_ag)
        for j in range(1, nstates_id)
            tax_prime[:, i, j] =
                (j .- 1.0) .* (mu .* wage_prime[:, i] .* u[i] ./ (1 .- u[i]))
        end
    end

    return iterate_policy(k_prime, K_prime, wealth, irate_prime, wage_prime, tax_prime, P)
end



function maketransition(a_prime, K, Z, distr, Π)
    # Generating grids
    alpha, beta, gamma, delta, mu, l_bar, k_ss = gen_params()
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
    nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()

    # Setup the distribution
    dPrime = zeros(eltype(distr), size(distr))

    # What is the current policy function given the realization of the aggregate state?
    # a_prime = reshape(k_prime, (ngridk, ngridkm, nstates_ag, nstates_id))
    a_cur_z = a_prime[:, :, Z, :]

    # Interpolating over the capital grid onto the current realization of the capital stock
    nodes = (k, km, epsilon)
    itp = interpolate(nodes, a_cur_z, Gridded(Linear()))
    extrp = extrapolate(itp, Flat())
    a_prime_t = extrp(k, [K], epsilon)
    # a_prime_t = itp(k, K, epsilon)

    # Creating interpolation bounds
    idm_n, wR_m_n = MakeWeightsLight(a_prime_t, k)
    blockindex = (0:nstates_id-1) * ngridk
    @views @inbounds begin
        for yy = 1:nstates_id # all current income states
            for aa = 1:ngridk
                dd = distr[aa, yy]
                IDD_n = idm_n[aa, yy]
                DL_n = (dd .* (1.0 .- wR_m_n[aa, yy]))
                DR_n = (dd .* wR_m_n[aa, yy])
                pp = (Π[2*(Z-1)+yy, :])
                for yy = 1:nstates_id
                    id_n = IDD_n .+ blockindex[yy]
                    dPrime[id_n] += (pp[yy] + pp[yy+2]) .* DL_n
                    dPrime[id_n+1] += (pp[yy] + pp[yy+2]) .* DR_n
                end
            end
        end
    end
    return dPrime
end


# Compute aggregate state
function aggregate_st(distr, k_prime, ag_shock)
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
    nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()

    # Initialize container
    km_ts = zeros(T)
    km_ts[1] = sum(sum(distr, dims = 2) .* k) # aggregate capital in t=1

    # Reshaping k_prime for the function
    k_star = reshape(k_prime, (ngridk, ngridkm, nstates_ag, nstates_id))

    for t in range(1, length = T - 1)
        dPrime = maketransition(k_star, km_ts[t], ag_shock[t], distr, prob)
        km_ts[t+1] = sum(sum(dPrime, dims = 2) .* k) # aggregate capital in t+1
        distr = dPrime
    end
    return km_ts, distr
end

# Function that generates an initial guess for the distribution
function F_distr(distr, grid, target_mean)

    # Output for two residuals
    residuals = zeros(3)

    # Condition one
    residuals[1] = sum(distr .* grid) - target_mean

    # Condition two
    residuals[2] = sum(distr) - 1.0

    # Condition three
    residuals[3] = sum(distr .< 0.0) + sum(distr .> 1.0)

    return residuals
end


# Solve it!
function solve_ALM(plotting = false, plotting_check = false)
    # generate shocks, grid, parameters, and convergence parameters
    id_shock, ag_shock = shocks()
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
    alpha, beta, gamma, delta, mu, l_bar, k_ss = gen_params()
    nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()
    B, dif_B, criter_k, criter_B, update_k, update_B = convergence_parameters()

    # Initial guess for the policy function
    k_prime = 0.9 * k
    n = ngridk * ngridkm * nstates_ag * nstates_id
    k_prime = reshape(k_prime, (length(k_prime), 1, 1, 1))
    k_prime = ones((ngridk, ngridkm, nstates_ag, nstates_id)) .* k_prime
    k_prime = reshape(k_prime, n)
    km_ts = zeros(T)
    c = zeros(n)

    # Finding a valid initial distribution
    println("Finding an initial distribution")
    distr = zeros((ngridk, nstates_id))
    f(x) = F_distr(x, k, k_ss)
    sol = nlsolve(f, distr)
    distr = sol.zero / sum(sol.zero)
    println("Initial distribution found")
    println(" ")

    """
    Main loop
    Solve for HH problem given ALM
    Generate time series km_ts given policy function
    Run regression and update ALM
    Iterate until convergence
    """
    iteration = 1
    while dif_B > criter_B && iteration < 100
        # Solve for HH policy functions at a given law of motion
        k_prime1, c = individual(k_prime, B)

        # Save difference in policy functions
        dif_pol = norm(k_prime1 - k_prime)
        k_prime = copy(k_prime1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k_prime, ag_shock)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = log.(km_ts[burn_in:end-1])[:]
        X = Array(
            [ones(length(x)) (ag_shock.-1)[burn_in:end-1] x (ag_shock.-1)[burn_in:end-1] .*
                                                            x],
        )
        y = log.(km_ts[(burn_in+1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y) 
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # if iteration % 100 == 0
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[burn_in:end]), digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        # println("Pure coefficients: ", B_new)
        # println("Coefficients: ", B_mat)
        println("R²: ", round(r2(ols), digits = 5))
        println(" ")
        # end

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """
        # if dif_B > (criter_B*100)
        distr = copy(distr1) # replace cross-sectional capital distribution
        # end

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = zeros(T)
            k_alm[1] = km_ts[1]
            for t in range(2, length = T - 1)
                k_alm[t] = exp(B[ag_shock[t-1], 1] .+ B[ag_shock[t-1], 2] * log(km_ts[t-1]))
            end
            plot(km_ts[end-200:end], label = "Model")
            plot!(k_alm[end-200:end], label = "ALM")
            display(plot!(title = "Capital series", xlabel = "Time", ylabel = "Capital"))
        end
        B = B_mat .* update_B .+ B .* (1 .- update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = zeros(T)
    k_alm[1] = km_ts[1]
    for t in range(2, length = T - 1)
        k_alm[t] = exp(B[ag_shock[t-1], 1] .+ B[ag_shock[t-1], 2] * log(km_ts[t-1]))
    end
    if plotting
        # Plotting the results
        plot(km_ts, label = "Realized")
        plot!(k_alm, label = "Forecasted")
        display(plot!(title = "Capital series", xlabel = "Time", ylabel = "Capital"))

        println("The norm between the two series is: ", norm(km_ts .- k_alm))
    end
    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k_prime, (ngridk, ngridkm, nstates_ag, nstates_id)),
    reshape(c, (ngridk, ngridkm, nstates_ag, nstates_id)),
    id_shock,
    ag_shock
end

# Solving the Krusell-Smith model
B, km_ts, k_pred, distr, k_prime, c, id_shock, ag_shock = @time solve_ALM(true, true);

# Compare coefficients between codes
