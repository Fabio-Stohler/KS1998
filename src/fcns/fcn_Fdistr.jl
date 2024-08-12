# Residual of a distribution
function F_distr(input::Array, grid::Array, target_mean::Float64)
    # Transform the input directly into a probability distribution
    distr = abs.(input) ./ sum(abs.(input))  # Ensures non-negative probabilities summing to 1

    # Calculate the residuals
    residual = sum(distr .* grid) - target_mean

    return [residual]
end

# Finding a valid initial distribution
function initial_distr(ngridk::Int64, nstates_id::Int64, asset_grid::Array, asset_target::Float64)
    input = rand(ngridk, nstates_id)  # Start with random positive numbers
    f(x) = F_distr(x, asset_grid, asset_target)

    sol = nlsolve(f, input, method=:trust_region, iterations=400)  # Method choice can impact convergence

    if KS.norm(f(sol.zero)) > 1e-5
        println("The initial distribution has an error of:")
        println(f(sol.zero)[:])
    end

    # Return the probability distribution
    distr = abs.(sol.zero) ./ sum(abs.(sol.zero))
    return distr
end