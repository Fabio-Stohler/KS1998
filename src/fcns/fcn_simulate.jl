# Compute aggregate state
function aggregate_st(
    distr::Array,
    k_prime::Array,
    ag_shock::Array,
    npar::Union{
        NumericalParameters,
        NumericalParametersBeta,
        NumericalParametersDelta,
        NumericalParametersAll,
    },
)
    # Initialize container
    km_ts = zeros(npar.T)
    km_ts[1] = sum(sum(distr; dims = 2) .* npar.k) # aggregate capital in t=1

    # Reshaping k_prime for the function
    k_star = reshape(k_prime, (npar.ngridk, npar.ngridkm, npar.nstates_id, npar.nstates_ag))
    @inbounds @views begin
        for t in range(1; length = npar.T - 1)
            dPrime =
                maketransition(k_star, km_ts[t], ag_shock[t], ag_shock[t + 1], distr, npar)
            km_ts[t + 1] = sum(sum(dPrime; dims = 2) .* npar.k) # aggregate capital in t+1
            distr = dPrime
        end
    end
    return km_ts, distr
end

# Function that computes time series of all aggregate variables
function simulate_aggregates(
    distr::Array,
    c_prime::Array,
    k_prime::Array,
    ag_shock::Array,
    mpar::ModelParameters,
    npar::Union{
        NumericalParameters,
        NumericalParametersBeta,
        NumericalParametersDelta,
        NumericalParametersAll,
    },
)
    # Initialize container
    K_ts = zeros(npar.T)
    C_ts = zeros(npar.T)
    I_ts = zeros(npar.T)
    N_ts = zeros(npar.T)
    Y_ts = zeros(npar.T)
    D_ts = Array{Array{eltype(distr)}}(undef, npar.T, 1)

    # aggregate capital in t=1
    K_ts[1] = sum(sum(distr; dims = 2) .* npar.k)

    # Reshaping k_prime for the function
    k_star = reshape(k_prime, (npar.ngridk, npar.ngridkm, npar.nstates_id, npar.nstates_ag))
    @inbounds @views begin
        for t in range(1; length = npar.T - 1)
            # Calculating the distribution in the next period
            dPrime =
                maketransition(k_star, K_ts[t], ag_shock[t], ag_shock[t + 1], distr, npar)

            # Interpolating the consumption policy function onto todays capital stock
            nodes = (npar.k, npar.km, npar.ϵ)
            itp = interpolate(nodes, c_prime[:, :, :, ag_shock[t]], Gridded(Linear()))
            extrp = extrapolate(itp, Flat())
            c_star = extrp(npar.k, [K_ts[t]], npar.ϵ)

            # Saving aggregates
            K_ts[t + 1] = sum(sum(dPrime; dims = 2) .* npar.k) # aggregate capital in t+1
            C_ts[t] = sum(sum(distr; dims = 2) .* c_star[:, :]) # aggregate consumption
            I_ts[t] = K_ts[t + 1] - (1 - mpar.δ) * K_ts[t] # aggregate investment
            N_ts[t] = [npar.er_b, npar.er_g][ag_shock[t]] .* mpar.l_bar # aggregate labor
            Y_ts[t] = npar.a[ag_shock[t]] * K_ts[t]^mpar.α * N_ts[t]^(1 - mpar.α) # aggregate output
            D_ts[t] = distr

            # Updating the distribution
            distr = dPrime
        end
    end

    # Last period
    # Interpolating the consumption policy function onto todays capital stock
    nodes = (npar.k, npar.km, npar.ϵ)
    itp = interpolate(nodes, c_prime[:, :, :, ag_shock[end]], Gridded(Linear()))
    extrp = extrapolate(itp, Flat())
    c_star = extrp(npar.k, K_ts[end], npar.ϵ)

    # Saving aggregates
    C_ts[end] = sum(sum(distr; dims = 2) .* c_star[:, 1, :]) # aggregate consumption
    N_ts[end] = [npar.er_b, npar.er_g][ag_shock[end]] .* mpar.l_bar # aggregate labor
    Y_ts[end] = npar.a[ag_shock[end]] * K_ts[end]^mpar.α * N_ts[end]^(1 - mpar.α) # aggregate output
    I_ts[end] = Y_ts[end] - C_ts[end] # aggregate investment
    D_ts[end] = distr

    return K_ts, C_ts, I_ts, N_ts, Y_ts, D_ts
end

# simulate and save the simulated data
function simulate_and_save(
    distr::Array,
    c_prime::Array,
    k_prime::Array,
    ag_shock::Array,
    mpar::ModelParameters,
    npar::Union{
        NumericalParameters,
        NumericalParametersBeta,
        NumericalParametersDelta,
        NumericalParametersAll,
    },
    name = "simulated_data",
    path = "../bld/data/",
)
    # simulate data
    K, C, I, N, Y, D = simulate_aggregates(distr, c_prime, k_prime, ag_shock, mpar, npar)

    # Create a tupple
    data = (K = K, C = C, I = I, N = N, Y = Y, D = D)

    # save data
    FileIO.save(path * name * ".jld2", "data", data)
end
