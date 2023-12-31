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
B, km_ts, k_pred, distr, k_prime, c, ag_shock = @time solve_ALM();