# Residual of a distribution
function F_distr(
        distr::Array, 
        grid::Array, 
        target_mean::Float64,
    )

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


# Finding a valid initial distribution
function initial_distr(ngridk::Int64, nstates_id::Int64, k::Array, k_ss::Float64)
    distr = zeros((ngridk, nstates_id))
    f(x) = F_distr(x, k, k_ss)
    sol = nlsolve(f, distr)
    distr = sol.zero / sum(sol.zero)
    return distr
end