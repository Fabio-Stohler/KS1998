# Function that generates an initial guess for the distribution
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