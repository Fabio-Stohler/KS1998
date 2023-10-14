"""
Solution to Krusell and Smith (1998) written by Fabio Stohler in Julia
"""

# setting the correct working directory
cd("./src")

# loading packages
push!(LOAD_PATH, pwd())
using KS, Revise

# Solving for the aggregate law of motion
using ProfileView
B, km_ts, k_pred, distr, k_prime, c, ag_shock = @time solve_ALM();


# Body of the solve_ALM_BK functions
# generate structures
mpar = ModelParameters()
npar = NumericalParameters()

# Initial guess for the ALM coefficients
A = zeros(npar.nstates_ag, npar.nstates_ag)
A[:, 1] .= 0.6
A[2, 1] += 0.05
A[:, 2] .= 0.1

B = zeros(npar.nstates_ag, npar.nstates_ag)
B[:, 1] .= 0.0
B[:, 2] .= 1.0

# Update the meshes for the computation here
KS.@set! npar.mesh_b = repeat(reshape(npar.b, (npar.ngridb, 1, 1, 1, 1)), outer = [1, npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id])
KS.@set! npar.mesh_k = repeat(reshape(npar.k, (1, npar.ngridk, 1, 1, 1)), outer = [npar.ngridb, 1, npar.ngridkm, npar.nstates_ag, npar.nstates_id])
KS.@set! npar.mesh_km = repeat(reshape(npar.km, (1, 1, npar.ngridkm, 1, 1)), outer = [npar.ngridb, npar.ngridk, 1, npar.nstates_ag, npar.nstates_id])
KS.@set! npar.mesh_a = repeat(reshape(npar.a, (1, 1, 1, npar.nstates_ag, 1)), outer = [npar.ngridb, npar.ngridk, npar.ngridkm, 1, npar.nstates_id])
KS.@set! npar.mesh_ϵ = repeat(reshape(npar.ϵ, (1, 1, 1, 1, npar.nstates_id)), outer = [npar.ngridb, npar.ngridk, npar.ngridkm, npar.nstates_ag, 1])

# Inputs to solve_HH
L = repeat(
    reshape([npar.er_b, npar.er_g], (1, 1, 1, npar.nstates_ag, 1)),
    outer = [npar.ngridb, npar.ngridk, npar.ngridkm, 1, npar.nstates_id],
) # Mesh over employment in states
r = 1.0 .+ KS.interest(npar.mesh_km, npar.mesh_a, L .* mpar.l_bar, mpar)
w = KS.wage(npar.mesh_km, npar.mesh_a, L .* mpar.l_bar, mpar)

# Guess for the state price
Qminus = zeros(npar.ngridb, npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id)
for km in 1:npar.ngridkm
    for z in 1:npar.nstates_ag
        Qminus[:, :, km, z, :] .= A[z, 1] .+ A[z, 2] .* log.(npar.km[km])
    end
end
Qminus[1, 1, :, :, 1]

# Defining income
inc::Array = [
    w .* npar.mesh_ϵ .* mpar.l_bar .+ mpar.μ .* (w .* (1 .- npar.mesh_ϵ)) .-
    mpar.μ .* (w .* (1 .- L) ./ L) .* npar.mesh_ϵ, # labor income
    npar.mesh_b, # bond income
    r .* npar.mesh_k, # capital income
]

res = @time KS.solve_HH(A, B, inc, r, npar, mpar, true);
res[1][:, :, :, :, 1]
res[2][:, :, :, :, 1]
res[2][:, :, :, :, 2]
res[3][:, :, :, :, 1]
res[4][:, :, :, :, 1]
res[5][:, :, :, :, 1]
