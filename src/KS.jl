# __precompile__(false)

module KS

using QuantEcon, Plots
using Random, Distributions, Dierckx
using Interpolations, GLM
using LinearAlgebra, Statistics
using NLsolve, FieldMetadata
using Parameters

export ModelParameters, 
    NumericalParameters,
    solve_ALM, 
    aggregate_st

# Including all necessary functions
include("./fcns/include_fcns.jl")

# Solve for the aggregate law of motion
function solve_ALM(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParameters()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, npar.nstates_ag)
    B[:, 1] .= 0.0
    B[:, 2] .= 1.0

    # Initial guess for the policy function
    n = npar.ngridk * npar.ngridkm * npar.nstates_ag * npar.nstates_id
    k_prime = reshape(0.9 * npar.k, (length(npar.k), 1, 1, 1))
    k_prime = ones((npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)) .* k_prime
    k_prime = reshape(k_prime, n)
    km_ts = zeros(npar.T)
    c = zeros(n)

    # Finding a valid initial distribution
    println("Finding an initial distribution")
    distr = zeros((npar.ngridk, npar.nstates_id))
    f(x) = F_distr(x, npar.k, mpar.k_ss)
    sol = nlsolve(f, distr)
    distr = sol.zero / sum(sol.zero)
    println("Initial distribution found")
    println(" ")

    """
    Main loop
    Solve for HH problem given ALM
    Generate time series km_ts given policy function
    Run regression and update ALM
    Iterate until convergence
    """
    iteration = 1
    dif_B = 10^10
    while dif_B > npar.ϵ_B && iteration < npar.iter_max
        # Solve for HH policy functions at a given law of motion
        k_prime1, c = individual(k_prime, B, mpar, npar)

        # Save difference in policy functions
        dif_pol = norm(k_prime1 - k_prime)
        k_prime = copy(k_prime1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k_prime, npar.ag_shock, npar)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = log.(km_ts[npar.burn_in:end-1])[:]
        X = Array(
            [ones(length(x)) (npar.ag_shock.-1)[npar.burn_in:end-1] x (npar.ag_shock.-1)[npar.burn_in:end-1] .*
                                                            x],
        )
        y = log.(km_ts[(npar.burn_in+1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y) 
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # if iteration % 100 == 0
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[npar.burn_in:end]), digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        # println("Pure coefficients: ", B_new)
        # println("Coefficients: ", B_mat)
        println("R²: ", round(r2(ols), digits = 5))
        println(" ")
        # end

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """
        # if dif_B > (criter_B*100)
        distr = copy(distr1) # replace cross-sectional capital distribution
        # end

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = zeros(npar.T)
            k_alm[1] = km_ts[1]
            for t in range(2, length = npar.T - 1)
                k_alm[t] = exp(B[npar.ag_shock[t-1], 1] .+ B[npar.ag_shock[t-1], 2] * log(k_alm[t-1]))
            end
            plot(km_ts, label = "Model")
            plot!(k_alm, label = "ALM")
            display(plot!(title = "Capital series", xlabel = "Time", ylabel = "Capital"))
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = zeros(npar.T)
    k_alm[1] = km_ts[1]
    for t in range(2, length = npar.T - 1)
        k_alm[t] = exp(B[npar.ag_shock[t-1], 1] .+ B[npar.ag_shock[t-1], 2] * log(k_alm[t-1]))
    end
    if plotting
        # Plotting the results
        plot(km_ts, label = "Model")
        plot!(k_alm, label = "ALM")
        display(plot!(title = "Capital series", xlabel = "Time", ylabel = "Capital"))

        println("The norm between the two series is: ", norm(km_ts .- k_alm))
    end
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k_prime, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock
end

end