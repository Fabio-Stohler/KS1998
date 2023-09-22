"""
Solution to Krusell and Smith (1998) written by Fabio Stohler in Julia
"""

# setting the correct working directory
cd("./src")

# loading packages
push!(LOAD_PATH, pwd())
using KS, Revise

# Solving for the aggregate law of motion
B, km_ts, k_pred, distr, k_prime_init, c_init, ag_shock = @time solve_ALM(true, true);
