# Residual of a distribution
function F_distr(input::Array, grid::Array, target_mean::Float64)
    # Transforms the input into a probability distribution
    N = Normal(0.0, 1.0)
    distr = pdf.(N, input) ./ sum(pdf.(N, input))

    # Output for two residuals
    residuals = zeros(2)

    # Condition one
    # residuals = sum(distr .* grid) - target_mean
    residuals[1] = sum(distr .* grid) - target_mean

    # Condition two
    residuals[2] = sum(distr) - 1.0

    return residuals
end


# Finding a valid initial distribution
function initial_distr(ngridk::Int64, nstates_id::Int64, k::Array, k_ss::Float64)
    input = ones((ngridk, nstates_id))
    f(x) = F_distr(x, k, k_ss)
    sol = nlsolve(f, input, iterations = 400) # Mostly no more gain after 400 iterations
    if norm(f(sol.zero)) > 10e-5
        println("The initial distribution has an error of:")
        println(f(sol.zero)[:])
    end
    distr = pdf.(Normal(), sol.zero) ./ sum(pdf.(Normal(), sol.zero))
    return distr
end
