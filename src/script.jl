"""
Solution to Krusell and Smith (1998) from Fabio Stohler 
"""

# setting the correct working directory
cd("./src")

# importing all necessary libraries
using QuantEcon
using Random

# function to set all parameters
function gen_params()
    β = 0.99        # discount factor
    γ = 1.0         # utility function parameters
    α = 0.36        # share of capital in production function
    δ = 0.025       # depreciation rate
    μ = 0.15        # unemployment benefits as a share of the wage
    l_bar = 1 / 0.9 # time endowment; normalizes s.t. L_bad = 1
    k_ss = (α*β/(1-β*(1-δ)))^(1/(1-α))
    return α, β, γ, δ, μ, l_bar, k_ss
end

function gen_grid()
    N = 10000       # number of agents for stochastic simulation
    J = 1000        # number of grid points for stochastic simulation
    k_min = 0
    k_max = 1000
    burn_in = 100
    T = 1000 + burn_in
    ngridk = 100
    x = range(0, 0.5, ngridk)
    τ = 7
    y = (x ./ maximum(x)).^τ
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
    ur_b = 0.1              # unemployment rate in a bad aggregate state
    er_b = (1 - ur_b)       # employment rate in a bad aggregate state
    ur_g = 0.04             # unemployment rate in a good aggregate state
    er_g = (1 - ur_g)       # employment rate in a good aggregate state
    ϵ = range(0, nstates_id-1)
    δ_a = 0.01
    a = [1 - δ_a, 1 + δ_a]
    Π = Array([0.525 0.35 0.03125 0.09375;
    0.038889 0.836111 0.002083 0.122917; 0.09375 0.03125 0.291667 0.583333;
    0.009115 0.115885 0.024306 0.850694])
    return nstates_id, nstates_ag, ϵ, ur_b, er_b, ur_g, er_g, a, Π
end


function shocks()
    N, J, k_min, k_max, T, burn_in, k, km_min, km_max,  km, ngridk, ngridkm = gen_grid()
    nstates_id, nstates_ag, ϵ, ur_b, er_b, ur_g, er_g, a, Π = shocks_parameters()
    ag_shock = zeros(T,1)
    id_shock = zeros(T,N)
    Random.seed!(123)

    # Transition probabilities between aggregate states
    prob_ag = zeros((nstates_ag, nstates_ag))
    prob_ag[1, 1] = Π[1, 1] + Π[1, 2]
    prob_ag[2, 1] = 1 - prob_ag[1, 1] # bad state to good state
    prob_ag[2, 2] = Π[3, 3] + Π[3, 4]
    prob_ag[1, 2] = 1 - prob_ag[2, 2]
    P = Π ./ kron(prob_ag, ones(nstates_ag, nstates_ag))

    # Generate aggregate shocks
    mc = MarkovChain(prob_ag)
    ag_shock = simulate(mc, T, init=1) # start from the bad state

    # generate idiosyncratic shocks for all agents in the first period
    draw = rand(N)
    id_shock[1, :] = (draw .> ur_b) .+ 1.0  # set state to good if probability exceeds ur_b

    # generate idiosyncratic shocks for all agents starting in second period
    draw = Random.rand(T-1,N)
    for t = range(2, length=T-1)
        # Fix idiosyncratic transition matrix conditional on aggregate state
        transition = P[ag_shock[t-1]:2*ag_shock[t-1],
        ag_shock[t]:2*ag_shock[t]]
        transition_prob = [transition[Int(id_shock[t-1, i]), 
        Int(id_shock[t-1, i])] for i = range(1, length=N)]
        # Check this again!
        check = transition_prob>draw[t-1, :] # sign whether to remain in current state
        id_shock[t, :] = (id_shock[t-1, :]) .* check .+ (3 .- id_shock[t-1, :]) .* (1 .- check)
    end
    return id_shock, ag_shock
end


function convergence_parameters(nstates_ag=2)
    dif_B = 10^10 # difference between coefficients B of ALM on succ. iter.
    ϵ_k = 1e-8
    ϵ_B = 1e-6
    update_k = 0.77
    update_B = 0.3
    B = zeros(nstates_ag, nstates_ag)
    B[:,2] .= 1.0
    return B, dif_B, ϵ_k, ϵ_B, update_k, update_B
end

# Solve the individual problem
function individual(k_prime, B)
    
end
dif_B, criter_k, criter_B, update_k, update_B = convergence_parameters()
N, J, k_min, k_max, T, burn_in, k, km_min, km_max, km, ngridk, ngridkm = gen_grid()
alpha, beta, gamma, delta, mu, l_bar, k_ss = gen_params()
nstates_id, nstates_ag, epsilon, ur_b, er_b, ur_g, er_g, a, prob = shocks_parameters()
e = Array([er_b, er_g])
u = 1 .- e
replacement = Array([mu, l_bar]) #replacement rate of wage
#Tax rate depending on aggregate and idiosyncratic states

#Transition probabilities by current state (k,km, Z, eps) and future (Z', eps')
n = ngridk*ngridkm*nstates_ag*nstates_id
P = zeros((ngridk, ngridkm, nstates_ag, nstates_id, nstates_ag*nstates_id))
for z = range(1, length = nstates_ag*nstates_id)
    for i = range(1, length = nstates_ag)
        for j = range(1, length = nstates_id)
            # Check this again!
            P[:, :, i, j, z] = prob[2*(i-1)+j, z] * ones((ngridk, ngridkm))
        end
    end
end
P = reshape(P, (n, nstates_ag*nstates_id))

k_indices = zeros(n)
km_indices = zeros(n)
ag = zeros(n)
e_i = zeros(n)
Cart_Indices = CartesianIndices(Array(reshape(range(1, length=n),(ngridk, ngridkm, nstates_ag, nstates_id))))
for s_i = range(1, length = n)
    k_indices[s_i], km_indices[s_i], ag[s_i], e_i[s_i] = Tuple(Cart_Indices[s_i])
end
k_indices = Array([Int(x) for x in k_indices])
km_indices = Array([Int(x) for x in km_indices])
ag = Array([Int(x) for x in ag])
e_i = Array([Int(x) for x in e_i])

"""
Using indices, generate arrays for productivity, employment, aggregate
capital, individual capital
"""

Z = Array([a[Int(i)] for i in ag])
L = Array([e[Int(i)] for i in ag])
K = Array([km[Int(i)] for i in km_indices])
k_i = Array([k[Int(i)] for i in k_indices])
irate = alpha.*Z.*(K./(l_bar.*L)).^(alpha-1)
wage = (1-alpha).*Z.*(K./(l_bar.*L)).^alpha
wealth = irate .* k_i .+ (wage .* e_i) .* l_bar .+ mu .* (wage .* (1 .- e_i)) .+ (1 .- delta) .* k_i .- mu .* (wage .* (1 .- L)./L).*e_i

# Transition of capital depends on aggregate state
K_prime = Array([exp(B[ag[i],1] .+ B[ag[i], 2] * log(K[i])) for i in range(1, length=n)])

# restrict km_prime to fall into bounds
K_prime = min.(K_prime, km_max)
K_prime = max.(K_prime, km_min)

# Future interst rate and wage conditional on state (bad or good state)
irate = zeros((n, nstates_ag))
wage = zeros((n, nstates_ag))
for i = range(1,nstates_ag)
    irate[:, i] = alpha.*a[i].*((K_prime./(e[i].*l_bar)).^(alpha.-1))
    wage[:,i] = (1 .- alpha) .* a[i] .* ((K_prime/(e[i]*l_bar)).^alpha)
end

# Tax rate
tax = zeros((n, nstates_ag, nstates_id))
for i in range(1, nstates_ag)
    for j in range(1, nstates_id)
        tax[:, i, j] = (j .- 1.0) .*(mu .* wage[:, i] .* u[i]./ (1 .- u[i]))
    end
end

# Solving the individual problem
function iterate_policy()
    
end
