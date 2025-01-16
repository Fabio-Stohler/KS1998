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
B, km_ts, k_pred, distr, k_prime, c, ag_shock = @time solve_ALM(true, true);

# Figure 1: the aggregate law of motion
npar = NumericalParameters();
KS.plot(npar.km, exp.(B[1, 1] .+ B[1, 2] .* log.(npar.km)); label = "Bad times");
KS.plot!(npar.km, exp.(B[2, 1] .+ B[2, 2] .* log.(npar.km)); label = "Good times");
KS.plot!(npar.km, npar.km; label = "45 degree line", linestyle = :dash, color = :black);
display(
    KS.plot!(;
        title = "Aggregate Law of Motion",
        xlabel = "Capital today",
        ylabel = "Capital tomorrow",
        legend = :topleft,
    ),
);

# Figure 2: the policy function
plotend = 60
KS.plot(npar.k[1:plotend], k_prime[1:plotend, 3, 2, 2]; label = "Employed");
KS.plot!(npar.k[1:plotend], k_prime[1:plotend, 3, 1, 2]; label = "Unemployed");
KS.plot!(
    npar.k[1:plotend],
    npar.k[1:plotend];
    label = "45 degree line",
    linestyle = :dash,
    color = :black,
);
display(
    KS.plot!(;
        title = "Capital Policy Function",
        xlabel = "Wealth",
        ylabel = "Next periods capital",
        legend = :topleft,
    ),
);

# Calculation of a stochastic steady state
km_ts_sss, distr_sss = KS.aggregate_st(distr, k_prime, ones.(Integer, npar.T + 1), npar);
km_ts_new = [km_ts; km_ts_sss];
KS.plot(km_ts_new; label = "Capital");
KS.plot!(
    repeat([ModelParameters().k_ss], length(km_ts_new));
    label = "Deterministic Steady State",
    linestyle = :dash,
    color = :black,
);
display(KS.plot!(; title = "Stochastic Steady State", xlabel = "Time", ylabel = "Capital"));
