# Solving the individual problem
function iterate_policy(
    k_prime::Array,
    K_prime::Array,
    wealth::Array,
    irate::Array,
    wage::Array,
    tax::Array,
    P::Array,
    mpar::ModelParameters,
    npar::NumericalParameters,
)
    # Extracting necessary stuff
    replacement = Array([mpar.μ, mpar.l_bar]) #replacement rate of wage
    n = npar.ngridk * npar.ngridkm * npar.nstates_ag * npar.nstates_id

    # Convergence parameters
    dif_k = 1
    iter_k = 1
    iter_k_max = 20000
    while dif_k > npar.ϵ_k && iter_k < iter_k_max
        """
            interpolate policy function k'=k(k, km) in new points (k', km')
        """
        k2_prime = zeros((n, npar.nstates_ag, npar.nstates_id))
        c_prime = zeros((n, npar.nstates_ag, npar.nstates_id))
        mu_prime = zeros((n, npar.nstates_ag, npar.nstates_id))

        #reshape k_prime for interpolation
        k_prime_reshape =
            reshape(k_prime, (npar.ngridk, npar.ngridkm, npar.nstates_id, npar.nstates_ag))
        K_prime_reshape =
            reshape(K_prime, (npar.ngridk, npar.ngridkm, npar.nstates_id, npar.nstates_ag))
        @inbounds @views begin
            for i in range(1, npar.nstates_ag)
                for j in range(1, npar.nstates_id)
                    # capital in aggregate state i, idiosyncratic state j as a function of current states
                    k2_prime[:, i, j] = reshape(
                        evaluate(
                            Spline2D(npar.k, npar.km, k_prime_reshape[:, :, i, j]),
                            k_prime_reshape[:],
                            K_prime_reshape[:],
                        ),
                        n,
                    )
                    c_prime[:, i, j] = (
                        irate[:, i] .* k_prime .+ replacement[j] .* (wage[:, i]) .+
                        (1 .- mpar.δ) .* k_prime .- k2_prime[:, i, j] .- tax[:, i, j]
                    )
                end
            end
        end

        # replace negative consumption by very low positive number
        c_prime = max.(c_prime, 10^(-10))
        mu_prime = c_prime .^ (-mpar.γ)

        #Expectation term in Euler equation
        #Components in terms of all possible transitions
        expec_comp = zeros((n, npar.nstates_ag, npar.nstates_id))
        @inbounds @views begin
            for i in range(1, length = npar.nstates_ag)
                for j in range(1, length = npar.nstates_id)
                    expec_comp[:, i, j] =
                        (mu_prime[:, i, j] .* (1 .- mpar.δ .+ irate[:, i])) .*
                        P[:, 2*(i-1)+j]
                end
            end
        end
        """
        Expectation term in the Euler equation
        sum over various transitions (which have already been scaled by their probability)
        """
        expec = sum(
            expec_comp[:, i, j] for i in range(1, length = npar.nstates_ag) for
            j in range(1, length = npar.nstates_id)
        )

        # current consumption from Euler equation if borrowing constraint is not binding
        cn = (mpar.β .* expec) .^ (-1 / mpar.γ)
        k_prime_n = wealth - cn
        k_prime_n = min.(k_prime_n, npar.k_max)
        k_prime_n = max.(k_prime_n, npar.k_min)
        """
        difference between new and previous capital functions
        """
        dif_k = norm(k_prime_n - k_prime)
        k_prime = npar.update_k .* k_prime_n .+ (1 .- npar.update_k) .* k_prime  # update k_prime_n
        iter_k += 1
    end
    if iter_k == iter_k_max
        println("EGM did not converge with error: ", dif_k)
    end
    c = wealth - k_prime
    return k_prime, c
end
