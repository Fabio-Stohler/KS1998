# Solve the individual problem
function individual(
    k_prime::Array,
    B::Array,
    mpar::ModelParameters,
    npar::NumericalParameters,
)
    # Extracting employment and unemployment rates
    e = Array([npar.er_b, npar.er_g])
    u = 1 .- e

    #Transition probabilities by current state (k,km, Z, eps) and future (Z', eps')
    n = npar.ngridk * npar.ngridkm * npar.nstates_ag * npar.nstates_id
    P = zeros((
        npar.ngridk,
        npar.ngridkm,
        npar.nstates_ag,
        npar.nstates_id,
        npar.nstates_ag * npar.nstates_id,
    ))
    @inbounds @views begin
        for z in range(1, length = npar.nstates_ag * npar.nstates_id)
            for i in range(1, length = npar.nstates_ag)
                for j in range(1, length = npar.nstates_id)
                    # Check this again!
                    P[:, :, i, j, z] =
                        npar.Π[2*(i-1)+j, z] * ones((npar.ngridk, npar.ngridkm))
                end
            end
        end
    end
    P = reshape(P, (n, npar.nstates_ag * npar.nstates_id))

    k_indices = zeros(n)
    km_indices = zeros(n)
    ag = zeros(n)
    e_i = zeros(n)
    Cart_Indices = CartesianIndices(
        Array(
            reshape(
                range(1, length = n),
                (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id),
            ),
        ),
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

    Z = Array([npar.a[Int(i)] for i in ag])
    L = Array([e[Int(i)] for i in ag])
    K = Array([npar.km[Int(i)] for i in km_indices])
    k_i = Array([npar.k[Int(i)] for i in k_indices])
    irate = mpar.α .* Z .* (K ./ (mpar.l_bar .* L)) .^ (mpar.α - 1)
    wage = (1 - mpar.α) .* Z .* (K ./ (mpar.l_bar .* L)) .^ mpar.α
    wealth =
        irate .* k_i .+ (wage .* e_i) .* mpar.l_bar .+ mpar.μ .* (wage .* (1 .- e_i)) .+
        (1 .- mpar.δ) .* k_i .- mpar.μ .* (wage .* (1 .- L) ./ L) .* e_i

    # Transition of capital depends on aggregate state
    K_prime =
        Array([exp(B[ag[i], 1] .+ B[ag[i], 2] * log(K[i])) for i in range(1, length = n)])

    # restrict km_prime to fall into bounds
    K_prime = min.(K_prime, npar.km_max)
    K_prime = max.(K_prime, npar.km_min)

    # Future interst rate and wage conditional on state (bad or good state)
    irate_prime = zeros((n, npar.nstates_ag))
    wage_prime = zeros((n, npar.nstates_ag))
    @inbounds @views begin
        for i in range(1, npar.nstates_ag)
            irate_prime[:, i] =
                mpar.α .* npar.a[i] .* ((K_prime ./ (e[i] .* mpar.l_bar)) .^ (mpar.α .- 1))
            wage_prime[:, i] =
                (1 .- mpar.α) .* npar.a[i] .* ((K_prime / (e[i] * mpar.l_bar)) .^ mpar.α)
        end
    end
    # Tax rate
    tax_prime = zeros((n, npar.nstates_ag, npar.nstates_id))
    @inbounds @views begin
        for i in range(1, npar.nstates_ag)
            for j in range(1, npar.nstates_id)
                tax_prime[:, i, j] =
                    (j .- 1.0) .* (mpar.μ .* wage_prime[:, i] .* u[i] ./ (1 .- u[i]))
            end
        end
    end

    return iterate_policy(
        k_prime,
        K_prime,
        wealth,
        irate_prime,
        wage_prime,
        tax_prime,
        P,
        mpar,
        npar,
    )
end
