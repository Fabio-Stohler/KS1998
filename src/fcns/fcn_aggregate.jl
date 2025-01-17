# Compute aggregate state
function aggregate_st(
    distr::Array,
    k_prime::Array,
    ag_shock::Array,
    npar::Union{NumericalParameters,NumericalParametersDelta,NumericalParametersBeta},
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
