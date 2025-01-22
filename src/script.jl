"""
    The following repository solves the Krusell and Smith, 1998 JPE economy with a perceived law of motion for the capital stock in the version with exogenous labor supply.

Fabio Stohler, University of Bonn, 22. November 2023
"""

# setting the correct working directory
cd("./src")

# loading packages
push!(LOAD_PATH, pwd())
using KS, Revise

# Solving for the aggregate law of motion
_, km_ts, k_pred, distr, k_prime, c, ag_shock = @time solve_ML(true, true);
B, km_ts, k_pred, distr, k_prime, c, ag_shock = @time solve_ALM(true, true);

# Figure 1: the aggregate law of motion
#npar = NumericalParameters();
#KS.plot(npar.km, exp.(B[1, 1] .+ B[1, 2] .* log.(npar.km)); label = "Bad times");
#KS.plot!(npar.km, exp.(B[2, 1] .+ B[2, 2] .* log.(npar.km)); label = "Good times");
#KS.plot!(npar.km, npar.km; label = "45 degree line", linestyle = :dash, color = :black);
#display(
#    KS.plot!(;
#        title = "Aggregate Law of Motion",
#        xlabel = "Capital today",
#        ylabel = "Capital tomorrow",
#        legend = :topleft,
#    ),
#);
#
## Figure 2: the policy function
#plotend = 60
#KS.plot(npar.k[1:plotend], k_prime[1:plotend, 3, 2, 2]; label = "Employed");
#KS.plot!(npar.k[1:plotend], k_prime[1:plotend, 3, 1, 2]; label = "Unemployed");
#KS.plot!(
#    npar.k[1:plotend],
#    npar.k[1:plotend];
#    label = "45 degree line",
#    linestyle = :dash,
#    color = :black,
#);
#display(
#    KS.plot!(;
#        title = "Capital Policy Function",
#        xlabel = "Wealth",
#        ylabel = "Next periods capital",
#        legend = :topleft,
#    ),
#);
#
## Plotting the distribution: All households are essentially like the representative agent
#KS.plot(distr; labels = ["Unemployed" "Employed"], color = [:red :blue]);
#display(KS.plot!(; title = "Distribution", xlabel = "Capital", ylabel = "Density"));
#
## Calculation of a stochastic steady state
#km_ts_sss, distr_sss = KS.aggregate_st(distr, k_prime, ones.(Integer, npar.T + 1), npar);
#km_ts_new = [km_ts; km_ts_sss];
#KS.plot(km_ts_new; label = "Capital");
#KS.plot!(
#    repeat([ModelParameters().k_ss], length(km_ts_new));
#    label = "Deterministic Steady State",
#    linestyle = :dash,
#    color = :black,
#);
#display(KS.plot!(; title = "Stochastic Steady State", xlabel = "Time", ylabel = "Capital"));

# importing required packages
using Flux, KS
using Flux.Optimise

# import the relevant parameters
npar = NumericalParameters();
mpar = ModelParameters();
mlpar = MLParameters();

# Generate a ML model
model = KS.generate_model(mpar, mlpar)

# collect parameters and Save
weights = collect(Flux.trainable(model))

# create the data
data = KS.create_data(km_ts[(npar.burn_in):end], npar.ag_shock[(npar.burn_in):end], mpar);

# predict some data
pred_temp = Float32.(KS.predict(model, data[1][1], mpar))

# define the loss function
loss(model, x, y) = Flux.mse(model(x), y'; agg = sum)

# define the optimizer
opt = Flux.setup(ADAM(0.001), model)

# create a vector to store the losses
losses = Float64[]
push!(losses, loss(model, data[1][1], data[1][2]))

# train the neural network
iter = 0
@time while iter < 2000 || losses[end] < 1e-6
    iter += 1
    Flux.train!(loss, model, data, opt)
    push!(losses, loss(model, data[1][1], data[1][2]))
    if iter % 250 == 0
        println("Iteration: $iter, Loss: $(losses[end])")
        # plot the loss
        # display(
        # KS.plot(
        # losses;
        # title = "Loss",
        # xlabel = "Iteration",
        # ylabel = "Log. Loss",
        # yscale = :log10,
        # ),
        # )

        # Generate time series of capital
        k_alm = zeros(npar.T)
        k_alm[1] = km_ts[1]
        for t in range(2; length = npar.T - 1)
            k_alm[t] = KS.predict(
                model,
                [KS.normalize_k(k_alm[t - 1], mpar), npar.ag_shock[t - 1]],
                mpar,
            )[1]
        end
        KS.plot(km_ts[(npar.burn_in):(end - 1)]; label = "Model")
        KS.plot!(k_alm[(npar.burn_in):(end - 1)]; label = "ALM")
        display(KS.plot!(; title = "Capital series", xlabel = "Time", ylabel = "Capital"))
    end
    if iter > 1000
        opt = Flux.setup(ADAM(0.00001), model)
    end
end
error = losses[end]

# plot the loss
display(
    KS.plot(
        losses;
        title = "Loss",
        xlabel = "Iteration",
        ylabel = "Log. Loss",
        yscale = :log10,
    ),
)

# extract new weights
weights_new = collect(Flux.trainable(model))

# combine the two models
model = KS.combine_models(model, Chain(weights...), 1.0)

# check if the weights are the same
weights_new == collect(Flux.trainable(model))
