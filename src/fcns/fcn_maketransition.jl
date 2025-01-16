function maketransition(
    a_prime::Array,
    K::Float64,
    Z::Int,
    Z_p::Int,
    distr::Array,
    npar::NumericalParameters,
)
    # Setup the distribution
    dPrime = zeros(eltype(distr), size(distr))

    # What is the current policy function given the realization of the aggregate state?
    # a_prime = reshape(k_prime, (ngridk, ngridkm, nstates_ag, nstates_id))
    a_cur_z = a_prime[:, :, :, Z]

    # Interpolating over the capital grid onto the current realization of the capital stock
    nodes = (npar.k, npar.km, npar.ϵ)
    itp = interpolate(nodes, a_cur_z, Gridded(Linear()))
    extrp = extrapolate(itp, Flat())
    a_prime_t = extrp(npar.k, [K], npar.ϵ)
    # a_prime_t = itp(k, K, npar.ϵ)

    # Creating interpolation bounds
    idm_n, wR_m_n = MakeWeightsLight(a_prime_t, npar.k)
    blockindex = (0:(npar.nstates_id - 1)) * npar.ngridk
    @views @inbounds begin
        for aa ∈ 1:(npar.ngridk) # all current capital states
            for yy ∈ 1:(npar.nstates_id) # all current income states
                dd = distr[aa, yy]
                IDD_n = idm_n[aa, yy]
                DL_n = (dd .* (1.0 .- wR_m_n[aa, yy]))
                DR_n = (dd .* wR_m_n[aa, yy])
                pp = (npar.Π[2 * (Z - 1) + yy, :])
                PP = (npar.Π_ag[Z, Z_p])
                for yy ∈ 1:(npar.nstates_id)
                    id_n = IDD_n .+ blockindex[yy]
                    dPrime[id_n] += (pp[Int(yy + 2 * (Z_p - 1))] / PP) .* DL_n
                    dPrime[id_n + 1] += (pp[Int(yy + 2 * (Z_p - 1))] / PP) .* DR_n
                end
            end
        end
    end
    if !(sum(dPrime) ≈ 1.0)
        println("Transition matrix does not sum to 1.0")
        println("Sum is: ", sum(dPrime))
    end
    return dPrime
end
