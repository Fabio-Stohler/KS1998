# __precompile__(false)

module KS

using QuantEcon, Plots
using Random, Distributions, Dierckx
using Interpolations, GLM
using LinearAlgebra, Statistics
using NLsolve, FieldMetadata
using Parameters, Setfield
using Flux, Flux.Optimise

export ModelParameters,
    NumericalParameters,
    MLParameters,
    solve_ALM,
    solve_ALM_Delta,
    solve_ALM_Beta,
    solve_ML,
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
        X = Array(
            [ones(length(x)) (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] x (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] .*
                                                                              x],
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # if iteration % 100 == 0
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        # println("Pure coefficients: ", B_new)
        # println("Coefficients: ", B_mat)
        println("R²: ", round(r2(ols); digits = 5))
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
            for t in range(2; length = npar.T - 1)
                k_alm[t] = exp(
                    B[npar.ag_shock[t - 1], 1] .+
                    B[npar.ag_shock[t - 1], 2] * log(k_alm[t - 1]),
                )
            end
            plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
            plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
            display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = zeros(npar.T)
    k_alm[1] = km_ts[1]
    for t in range(2; length = npar.T - 1)
        k_alm[t] = exp(
            B[npar.ag_shock[t - 1], 1] .+ B[npar.ag_shock[t - 1], 2] * log(k_alm[t - 1]),
        )
    end
    if plotting
        # Plotting the results
        plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
        plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
        display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))

        println(
            "The norm between the two series is: ",
            norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
        )
    end
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock
end

function solve_ALM_Delta(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParametersDelta()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, npar.nstates_ag)
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
        X = Array(
            [ones(length(x)) (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] x (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] .*
                                                                              x],
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # if iteration % 100 == 0
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        # println("Pure coefficients: ", B_new)
        # println("Coefficients: ", B_mat)
        println("R²: ", round(r2(ols); digits = 5))
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
            for t in range(2; length = npar.T - 1)
                k_alm[t] = exp(
                    B[npar.ag_shock[t - 1], 1] .+
                    B[npar.ag_shock[t - 1], 2] * log(k_alm[t - 1]),
                )
            end
            plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
            plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
            display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = zeros(npar.T)
    k_alm[1] = km_ts[1]
    for t in range(2; length = npar.T - 1)
        k_alm[t] = exp(
            B[npar.ag_shock[t - 1], 1] .+ B[npar.ag_shock[t - 1], 2] * log(k_alm[t - 1]),
        )
    end
    if plotting
        # Plotting the results
        plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
        plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
        display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))

        println(
            "The norm between the two series is: ",
            norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
        )
    end
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock
end

function solve_ALM_Beta(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParametersBeta()

    # Initial guess for the ALM coefficients
    B = zeros(npar.nstates_ag, npar.nstates_ag)
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
        X = Array(
            [ones(length(x)) (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] x (npar.ag_shock .- 1)[(npar.burn_in):(end - 1)] .*
                                                                              x],
        )
        y = log.(km_ts[(npar.burn_in + 1):end])[:]
        ols = lm(X, y)
        B_new = coef(ols)
        # B_new = inv(X'*X)*(X'*y)
        B_mat =
            reshape([B_new[1], B_new[1] + B_new[2], B_new[3], B_new[3] + B_new[4]], (2, 2))
        dif_B = norm(B_mat - B)

        # if iteration % 100 == 0
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", dif_B)
        println("Error in pol. fcn.: ", dif_pol)
        # println("Pure coefficients: ", B_new)
        # println("Coefficients: ", B_mat)
        println("R²: ", round(r2(ols); digits = 5))
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
            for t in range(2; length = npar.T - 1)
                k_alm[t] = exp(
                    B[npar.ag_shock[t - 1], 1] .+
                    B[npar.ag_shock[t - 1], 2] * log(k_alm[t - 1]),
                )
            end
            plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
            plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
            display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))
        end
        B = B_mat .* npar.update_B .+ B .* (1 .- npar.update_B) # update the vector of ALM coefficients
        iteration += 1
    end

    # Generate time series of capital
    k_alm = zeros(npar.T)
    k_alm[1] = km_ts[1]
    for t in range(2; length = npar.T - 1)
        k_alm[t] = exp(
            B[npar.ag_shock[t - 1], 1] .+ B[npar.ag_shock[t - 1], 2] * log(k_alm[t - 1]),
        )
    end
    if plotting
        # Plotting the results
        plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
        plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
        display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))

        println(
            "The norm between the two series is: ",
            norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
        )
    end
    println("The coefficients are:")
    println(B[1, 1], " ", B[1, 2])
    println(B[2, 1], " ", B[2, 2])

    return B,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock
end

# Solve for the aggregate law of motion
function solve_ML(plotting = false, plotting_check = false)
    # generate structures
    mpar = ModelParameters()
    npar = NumericalParameters()
    mlpar = MLParameters()

    # Generate a ML model
    model = generate_model(mpar, mlpar)

    # define the loss function
    loss(model, x, y) = Flux.mse(model(x), y'; agg = sum)

    # create a vector to store the losses
    losses = Float64[]
    lr = mlpar.lr

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
    error = 10^10
    while error > npar.ϵ_B && iteration < npar.iter_max
        # Solve for HH policy functions at a given law of motion
        c, k1 = solve_HH(model, inc, r_t, npar, mpar)

        # Save difference in policy functions
        dif_pol = norm(k1 - k0)
        k0 = copy(k1)

        # Generate time series and cross section of capital
        km_ts, distr1 = aggregate_st(distr, k1, npar.ag_shock, npar)

        # Plotting the results
        if plotting_check
            # Generate time series of capital
            k_alm = zeros(npar.T)
            k_alm[1] = km_ts[1]
            for t in range(2; length = npar.T - 1)
                k_alm[t] = predict(
                    model,
                    [normalize_k(k_alm[t - 1], mpar), npar.ag_shock[t - 1]],
                    mpar,
                )[1]
            end

            # plot the capital series
            plot1 = plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
            plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
            plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital")
        end

        # create the data
        data =
            create_data(km_ts[(npar.burn_in):end], npar.ag_shock[(npar.burn_in):end], mpar)

        # define the optimizer
        opt = Flux.setup(ADAM(lr), model)

        # Reduce the number of epochs after the first iteration
        if length(losses) > 0
            @set! mlpar.n_epochs = 1000
        end

        # set the learning rate to a lower value after 1000 iterations
        if length(losses) == mlpar.n_epochs
            @set! mlpar.lr = 0.0001
        end

        # train the neural network
        for epoch = 1:(mlpar.n_epochs)
            # update the learning rate
            if (length(losses) > 10000) & (length(losses) % 5000 == 0)
                lr = lr / 10
                opt = Flux.setup(ADAM(lr), model)
            elseif ((epoch % 1000 == 0) & (lr > 1e-6))
                lr = lr / 10
                opt = Flux.setup(ADAM(lr), model)
            end
            Flux.train!(loss, model, data, opt)
            push!(losses, loss(model, data[1][1], data[1][2]))
        end
        error = losses[end]

        if plotting_check
            # Generating a new time series with the trained network
            k_trained = zeros(npar.T)
            k_trained[1] = km_ts[1]
            for t in range(2; length = npar.T - 1)
                k_trained[t] = predict(
                    model,
                    [normalize_k(k_trained[t - 1], mpar), npar.ag_shock[t - 1]],
                    mpar,
                )[1]
            end
            plot!(k_trained[(npar.burn_in):(end - 1)]; label = "Trained")

            # plot the loss
            plot2 = plot(losses; label = "Loss", yscale = :log10)
            plot!(plot2; title = "Loss", xlabel = "Iteration", ylabel = "Log. Loss")

            # plot both plots in a single plot
            display(plot(plot1, plot2; layout = (2, 1), size = (900, 600)))
        end

        # if iteration % 100 == 0
        println("Iteration: ", iteration)
        println("Average capital: ", round(mean(km_ts[(npar.burn_in):end]); digits = 5))
        println("Error: ", error)
        println("Error in pol. fcn.: ", dif_pol)
        println("The learning rate is: ", lr)
        # println("Pure coefficients: ", B_new)
        println(" ")

        # To ensure that the initial capital distribution comes from the ergodic set,
        distr = copy(distr1) # replace cross-sectional capital distribution
        iteration += 1

        # update the model
        # model = combine_models(model, Chain(weights_old...), 0.2)
    end

    # Generate time series of capital
    k_alm = zeros(npar.T)
    k_alm[1] = km_ts[1]
    for t in range(2; length = npar.T - 1)
        k_alm[t] =
            predict(model, [normalize_k(k_alm[t - 1], mpar), npar.ag_shock[t - 1]], mpar)[1]
    end
    if plotting
        # Plotting the results
        plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
        plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
        display(plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))

        println(
            "The norm between the two series is: ",
            norm(km_ts[(npar.burn_in):end] .- k_alm[(npar.burn_in):end]),
        )
    end
    # println("The coefficients are:")
    # println(B[1, 1], " ", B[1, 2])
    # println(B[2, 1], " ", B[2, 2])

    return model,
    km_ts,
    k_alm,
    distr,
    reshape(k1, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    reshape(c, (npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)),
    npar.ag_shock
end

end
