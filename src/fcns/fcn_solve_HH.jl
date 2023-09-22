@doc raw"""
    solve_HH(B::Array, inc::Vector, r_t::Array, npar::NumericalParameters, mpar::ModelParameters, verbose::Bool = false)    

Solving the household problem via endogeneous grid methods.

Inputs:
- B::Array{Float64, 2}:             Perceived law of motion of the households
- inc::Vector{Array{Float64, 4}}:   Income process
- r_t::Array{Float64, 4}:           Interest rate
- npar::NumericalParameters:        Numerical parameters of the model
- mpar::ModelParameters:            Economic parameters of the model
- verbose::Bool:                    Print progress to screen

Outputs:
- c_star::Array:                         policy function for consumption
- k_star::Array:                    policy function for savings
"""
# Solve the individual problem
function solve_HH(
    B::Array{Float64,2},
    inc::Vector{Array{Float64,4}},
    r_t::Array{Float64,4},
    npar::NumericalParameters,
    mpar::ModelParameters,
    verbose::Bool = false,
)

    # Guess the initial value for rmu
    rmu = r_t .* mutil(inc[1] .+ (r_t .- 1.0) .* inc[2], mpar)

    # containers for policies, marginal value functions etc.
    k_star::Array{Float64,4} = similar(rmu)
    c_star::Array{Float64,4} = similar(rmu)
    EVk::Array{Float64,4} = similar(rmu)
    Vk::Array{Float64,5} =
        zeros(npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id, npar.nstates_ag)
    rmu_new::Array{Float64,4} = similar(rmu)
    irmu::Array{Float64,4} = invmutil(rmu, mpar)
    irmu_new::Array{Float64,4} = similar(irmu)
    EMU::Array{Float64,4} = similar(EVk)
    c_star_temp::Array{Float64,4} = similar(EVk)
    k_star_temp::Array{Float64,4} = similar(EVk)

    #   initialize distance variables
    count::Int = 0
    dist::Float64 = 9999.0

    while dist > npar.ϵ_k && count < npar.iter_max_k # Iterate consumption policies until converegence
        count = count + 1
        # Take expectations for labor income change
        update_EVk!(EVk, Vk, rmu, B, mpar, npar)

        # Policy update step
        EGM_policyupdate!(
            c_star,
            k_star,
            EMU,
            c_star_temp,
            k_star_temp,
            EVk,
            r_t,
            inc,
            npar,
            mpar,
            false,
        )

        # marginal value update step
        rmu_new = r_t .* mutil(c_star, mpar)
        invmutil!(irmu_new, rmu_new, mpar)
        dist = maximum(abs, irmu_new .- irmu)

        # update policy guess/marginal values of liquid/illiquid assets
        rmu .= rmu_new
        irmu .= irmu_new
        if verbose && count % 100 == 0
            println("EGM Iterations: ", count)
            println("EGM Dist: ", dist)
            println(" ")
        end
    end
    return c_star, k_star
end
