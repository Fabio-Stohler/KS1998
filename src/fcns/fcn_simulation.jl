function simulate_data(km_ts, B, npar)
    # Generate time series of capital
    k_alm = zeros(npar.T)
    k_alm[1] = km_ts[1]
    for t = 2:(npar.T)
        k_alm[t] =
            exp(B[npar.ag_shock[t - 1], 1] + B[npar.ag_shock[t - 1], 2] * log(km_ts[t - 1]))
    end
    return k_alm
end

function plot_simulation(km_ts, k_alm, label = "", path = "")
    # Plotting the time series of capital
    plot(km_ts[(end - 1001):(end - 1)]; label = "Model")
    plot!(k_alm[(end - 1001):(end - 1)]; label = "ALM")
    display(plot!(; title = "Capital series" * label, xlabel = "Time", ylabel = "Capital"))

    # Saving the plot
    if path != ""
        savefig(path * "Capital series" * label * ".png")
    end
end
