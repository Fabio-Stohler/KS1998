# __precompile__(false)

module KS

using QuantEcon, Plots
using Random, Distributions, Dierckx
using Interpolations, GLM
using LinearAlgebra, Statistics
using NLsolve, FieldMetadata
using Parameters, Setfield

export ModelParameters,
    NumericalParameters,
    NumericalParametersDelta,
    NumericalParametersBeta,
    NumericalParametersAll,
    solve_ALM,
    solve_ALM_Delta,
    solve_ALM_Beta,
    solve_ALM_All,
    aggregate_st

# Including all necessary functions
include("./fcns/include_fcns.jl")

# Solve for the aggregate law of motion
function solve_ALM(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParameters()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, 2)
    B[:, 1] .= 0.0
    B[:, 2] .= 1.0

    # Inputs to solve_HH
    L = repeat(
        reshape([npar.er_b, npar.er_g], (1, 1, 1, npar.nstates_ag));
        outer = [npar.ngridk, npar.ngridkm, npar.nstates_id, 1],
    ) # Mesh over employment in states
    r_t = 1.0 .+ KS.interest(npar.mesh_km, npar.mesh_a, L .* mpar.l_bar, mpar)
    w_t = KS.wage(npar.mesh_km, npar.mesh_a, L .* mpar.l_bar, mpar)

    # Defining income
    inc::Array = [
        w_t .* npar.mesh_ϵ .* mpar.l_bar .+ mpar.μ .* (w_t .* (1 .- npar.mesh_ϵ)) .-
        mpar.μ .* (w_t .* (1 .- L) ./ L) .* npar.mesh_ϵ,
        r_t .* npar.mesh_k,
    ]

    # Initial guess for the policy function
    km_ts = zeros(npar.T)
    c = similar(r_t)
    k0 = similar(r_t)
    k1 = similar(r_t)

    # Initial guess for the cross-sectional distribution
    distr = npar.distr_init

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
        c, k1 = solve_HH(B, inc, r_t, npar, mpar)

        # Save difference in policy functions
        dif_pol = norm(k1 - k0)
        k0 = copy(k1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k1, npar.ag_shock, npar)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = log.(km_ts[(npar.burn_in):(end - 1)])[:]
        X = hcat(
            ones(length(x)),
            (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)],
            x,
            (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] .* x,
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        println("R²: ", round(r2(ols); digits = 5))
        println(" ")

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """

        # replace cross-sectional capital distribution
        distr = copy(distr1)

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = simulate_data(km_ts, B, npar)

            # plot the results
            plot_simulation(km_ts, k_alm)
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = simulate_data(km_ts, B, npar)

    if plotting
        # Plotting the results
        plot_simulation(km_ts, k_alm, "../bld/figures/beta/")
    end
    println(
        "The norm between the two series is: ",
        norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
    )
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock,
    mpar,
    npar
end

function solve_ALM_Beta(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParametersBeta()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, 2)
    B[:, 1] .= 0.0
    B[:, 2] .= 1.0

    # Inputs to solve_HH
    L = repeat(
        reshape([npar.er_b, npar.er_g], (1, 1, 1, npar.nstates_ag));
        outer = [npar.ngridk, npar.ngridkm, npar.nstates_id, 1],
    ) # Mesh over employment in states
    r_t = 1.0 .+ KS.interest(npar.mesh_km, [1.0], L .* mpar.l_bar, mpar)
    w_t = KS.wage(npar.mesh_km, ones(size(npar.mesh_k)), L .* mpar.l_bar, mpar)

    # Defining income
    inc::Array = [
        w_t .* npar.mesh_ϵ .* mpar.l_bar .+ mpar.μ .* (w_t .* (1 .- npar.mesh_ϵ)) .-
        mpar.μ .* (w_t .* (1 .- L) ./ L) .* npar.mesh_ϵ,
        r_t .* npar.mesh_k,
    ]

    # Initial guess for the policy function
    km_ts = zeros(npar.T)
    c = similar(r_t)
    k0 = similar(r_t)
    k1 = similar(r_t)

    # Initial guess for the cross-sectional distribution
    distr = npar.distr_init

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
        c, k1 = solve_HH(B, inc, r_t, npar, mpar)

        # Save difference in policy functions
        dif_pol = norm(k1 - k0)
        k0 = copy(k1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k1, npar.ag_shock, npar)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = log.(km_ts[(npar.burn_in):(end - 1)])[:]
        X = hcat(
            ones(length(x)),
            (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)],
            x,
            (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] .* x,
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # Print statements
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        println("R²: ", round(r2(ols); digits = 5))
        println(" ")

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """

        # replace cross-sectional capital distribution
        distr = copy(distr1)

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = simulate_data(km_ts, B, npar)

            # plot the results
            plot_simulation(km_ts, k_alm)
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = simulate_data(km_ts, B, npar)

    if plotting
        # Plotting the results
        plot_simulation(km_ts, k_alm, "../bld/figures/beta/")
    end
    println(
        "The norm between the two series is: ",
        norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
    )
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock,
    mpar,
    npar
end

function solve_ALM_Delta(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParametersDelta()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, 2)
    B[:, 1] .= 0.0
    B[:, 2] .= 1.0

    # Inputs to solve_HH
    L = repeat(
        reshape([npar.er_b, npar.er_g], (1, 1, 1, npar.nstates_ag));
        outer = [npar.ngridk, npar.ngridkm, npar.nstates_id, 1],
    ) # Mesh over employment in states
    r_t = 1.0 .+ KS.interest(npar.mesh_km, 1.0, L .* mpar.l_bar, npar.mesh_δ, mpar)
    w_t = KS.wage(npar.mesh_km, ones(size(npar.mesh_k)), L .* mpar.l_bar, mpar)

    # Defining income
    inc::Array = [
        w_t .* npar.mesh_ϵ .* mpar.l_bar .+ mpar.μ .* (w_t .* (1 .- npar.mesh_ϵ)) .-
        mpar.μ .* (w_t .* (1 .- L) ./ L) .* npar.mesh_ϵ,
        r_t .* npar.mesh_k,
    ]

    # Initial guess for the policy function
    km_ts = zeros(npar.T)
    c = similar(r_t)
    k0 = similar(r_t)
    k1 = similar(r_t)

    # Initial guess for the cross-sectional distribution
    distr = npar.distr_init

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
        c, k1 = solve_HH(B, inc, r_t, npar, mpar)

        # Save difference in policy functions
        dif_pol = norm(k1 - k0)
        k0 = copy(k1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k1, npar.ag_shock, npar)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = log.(km_ts[(npar.burn_in):(end - 1)])[:]
        X = hcat(
            ones(length(x)),
            (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)],
            x,
            (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] .* x,
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # Print statements
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        println("R²: ", round(r2(ols); digits = 5))
        println(" ")

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """

        # replace cross-sectional capital distribution
        distr = copy(distr1)

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = simulate_data(km_ts, B, npar)

            # plot the results
            plot_simulation(km_ts, k_alm)
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = simulate_data(km_ts, B, npar)

    if plotting
        # Plotting the results
        plot_simulation(km_ts, k_alm, "../bld/figures/beta/")
    end
    println(
        "The norm between the two series is: ",
        norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
    )
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock,
    mpar,
    npar
end

function solve_ALM_All(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParametersAll()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, 2)
    B[:, 1] .= 0.0
    B[:, 2] .= 1.0

    # Inputs to solve_HH
    L = repeat(
        reshape(repeat([npar.er_b, npar.er_g], 4), (1, 1, 1, npar.nstates_ag));
        outer = [npar.ngridk, npar.ngridkm, npar.nstates_id, 1],
    ) # Mesh over employment in states
    r_t = 1.0 .+ KS.interest(npar.mesh_km, npar.mesh_a, L .* mpar.l_bar, npar.mesh_δ, mpar)
    w_t = KS.wage(npar.mesh_km, npar.mesh_a, L .* mpar.l_bar, mpar)

    # Defining income
    inc::Array = [
        w_t .* npar.mesh_ϵ .* mpar.l_bar .+ mpar.μ .* (w_t .* (1 .- npar.mesh_ϵ)) .-
        mpar.μ .* (w_t .* (1 .- L) ./ L) .* npar.mesh_ϵ,
        r_t .* npar.mesh_k,
    ]

    # Initial guess for the policy function
    km_ts = zeros(npar.T)
    c = similar(r_t)
    k0 = similar(r_t)
    k1 = similar(r_t)

    # Initial guess for the cross-sectional distribution
    distr = npar.distr_init

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
        c, k1 = solve_HH(B, inc, r_t, npar, mpar)

        # Save difference in policy functions
        dif_pol = norm(k1 - k0)
        k0 = copy(k1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k1, npar.ag_shock, npar)
        """
        run regression: log(km') = B[j,1]+B[j,2]log(km) for aggregate state
        """
        x = log.(km_ts[(npar.burn_in):(end - 1)])[:]
        X = hcat(
            ones(length(x)),
            (npar.a_shock .- 1)[(npar.burn_in):(end - 1)],
            (npar.β_shock .- 1)[(npar.burn_in):(end - 1)],
            (npar.δ_shock .- 1)[(npar.burn_in):(end - 1)],
            x,
            (npar.a_shock .- 1)[(npar.burn_in):(end - 1)] .* x,
            (npar.β_shock .- 1)[(npar.burn_in):(end - 1)] .* x,
            (npar.δ_shock .- 1)[(npar.burn_in):(end - 1)] .* x,
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat = reshape(
            [
                B_new[1],
                B_new[1] + B_new[2],
                B_new[1] + B_new[3],
                B_new[1] + B_new[2] + B_new[3],
                B_new[1] + B_new[4],
                B_new[1] + B_new[2] + B_new[4],
                B_new[1] + B_new[3] + B_new[4],
                B_new[1] + B_new[2] + B_new[3] + B_new[4],
                B_new[5],
                B_new[5] + B_new[6],
                B_new[5] + B_new[7],
                B_new[5] + B_new[6] + B_new[7],
                B_new[5] + B_new[8],
                B_new[5] + B_new[6] + B_new[8],
                B_new[5] + B_new[7] + B_new[8],
                B_new[5] + B_new[6] + B_new[7] + B_new[8],
            ],
            (npar.nstates_ag, 2),
        )
        dif_B = norm(B_mat - B)

        # Print statements
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        println("R²: ", round(r2(ols); digits = 5))
        println(" ")

        """
        To ensure that the initial capital distribution comes from the ergodic set,
        we use the terminal distribution of the current iteration as the initial distribution for
        subsequent iterations.

        When the solution is sufficiently accurate, we stop the updating and hold the distribution
        k_cross fixed for the remaining iterations
        """

        # replace cross-sectional capital distribution
        distr = copy(distr1)

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = simulate_data(km_ts, B, npar)

            # plot the results
            plot_simulation(km_ts, k_alm)
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = simulate_data(km_ts, B, npar)

    if plotting
        # Plotting the results
        plot_simulation(km_ts, k_alm, "../bld/figures/beta/")
    end
    println(
        "The norm between the two series is: ",
        norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
    )
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])
    println(B[3, 1], " ", B[3, 2])
    println(B[4, 1], " ", B[4, 2])
    println(B[5, 1], " ", B[5, 2])
    println(B[6, 1], " ", B[6, 2])
    println(B[7, 1], " ", B[7, 2])
    println(B[8, 1], " ", B[8, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock,
    mpar,
    npar
end

end
