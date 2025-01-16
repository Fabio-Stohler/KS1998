@doc raw"""
    EGM_policyupdate(EVk, r_minus, inc, npar, mpar, warnme)

Find optimal policies, given marginal continuation value `EVk`, today's interest rate `r_minus`, and income `inc`, using the Endogenous Grid Method.

Optimal policies are defined on the fixed grid, but optimal choices (`c` and `k`) are off-grid values.

# Inputs
- `EVk::Array{Float64, 4}`: marginal continuation value, defined on the fixed grid
- `r_minus::Array{Float64, 4}`: interest rate, defined on the fixed grid
- `inc::Array`: income, defined on the fixed grid
- `npar::NumericalParameters`: numerical parameters
- `mpar::ModelParameters`: model parameters
- `warnme::Bool`: whether to warn if non-monotonicity is encountered

# Returns
- `c_star::Array`,`k_star::Array`: optimal (on-grid) policies for
    consumption [`c`], and [`k`] asset
"""
function EGM_policyupdate(
    EVk::Array{Float64,4},
    r_minus::Array{Float64,4},
    inc::Array,
    npar::NumericalParameters,
    mpar::ModelParameters,
    warnme::Bool,
)

    # Pre-Allocate returns
    c_star = similar(EVk) # Initialize c-container
    k_star = similar(EVk) # Initialize m-container
    # containers for auxiliary variables
    EMU = similar(EVk)
    c_star_temp = similar(EVk)
    k_star_temp = similar(EVk)
    EGM_policyupdate!(
        c_star,
        k_star,
        EMU,
        c_star_temp,
        k_star_temp,
        EVk,
        r_minus,
        inc,
        npar,
        mpar,
        warnme,
    )

    return c_star, k_star
end

@doc raw"""
    EGM_policyupdate!(c_star, k_star, EMU, c_star_temp, k_star_temp, EVk, r_minus, inc, npar, mpar, warnme)

Find optimal policies, given marginal continuation value `EVk`, today's interest rate `r_minus`, and income `inc`, using the Endogenous Grid Method. EGM_policyupdate! is the same as EGM_policyupdate, but it modifies the inputs in-place.

# Inputs
- `c_star::Array{Float64, 4}`: optimal (on-grid) policies for consumption [`c`]
- `k_star::Array{Float64, 4}`: optimal (on-grid) policies for [`k`] asset
- `EMU::Array{Float64, 4}`: expected marginal utility, defined on the fixed grid
- `c_star_temp::Array{Float64, 4}`: optimal (off-grid) policies for consumption [`c`]
- `k_star_temp::Array{Float64, 4}`: optimal (off-grid) policies for [`k`] asset
- `EVk::Array{Float64, 4}`: marginal continuation value, defined on the fixed grid
- `r_minus::Array{Float64, 4}`: interest rate, defined on the fixed grid
- `inc::Array`: income, defined on the fixed grid
- `npar::NumericalParameters`: numerical parameters
- `mpar::ModelParameters`: model parameters
- `warnme::Bool`: whether to warn if non-monotonicity is encountered
"""
function EGM_policyupdate!(
    c_star::Array{Float64,4},
    k_star::Array{Float64,4},
    EMU::Array{Float64,4},
    c_star_temp::Array{Float64,4},
    k_star_temp::Array{Float64,4},
    EVk::Array{Float64,4},
    r_minus::Array{Float64,4},
    inc::Array,
    npar::NumericalParameters,
    mpar::ModelParameters,
    warnme::Bool,
)

    ################### Copy/read-out stuff ###################
    β::Float64 = mpar.β

    # inc[1] = labor income ,
    # inc[2] = assets income
    inc_lab = inc[1]
    inc_LA = inc[2]
    n = size(EVk)

    #############################################################
    ########## EGM Step 1: Find optimal asset holdings ##########
    #############################################################
    EMU .= EVk .* β
    KS.invmutil!(c_star_temp, EMU, mpar) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [k']
    # Calculate initial asset position from the budget constraint
    # that leads to the optimal consumption choice
    k_star_temp .= (c_star_temp .+ npar.mesh_k .- inc_lab)

    # Apply correct interest rate for the respective states
    k_star_temp .= k_star_temp ./ (r_minus)  # apply interest rate

    # Next step: Interpolate grid_k and c_star_temp from new k grid
    # Interpolate grid().k and c_n_aux defined on k_star_temp onto grid().k
    # Check monotonicity of m_star_n
    if warnme
        k_star_aux = reshape(k_star_temp, (n[1], n[2], n[3], n[4]))
        if any(any(diff(k_star_aux; dims = 1) .< 0))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies on grid tuples (k, km, z, e) are now given.
    # Need to interpolate to return to fixed grid.
    @inbounds @views begin
        for kk ∈ 1:n[2] # Loop over aggregate capital states
            for jj ∈ 1:n[3] # Loop over income states
                for zz ∈ 1:n[4] # Loop over productivity states
                    mylinearinterpolate_mult2!(
                        c_star[:, kk, jj, zz],
                        k_star[:, kk, jj, zz],
                        k_star_temp[:, kk, jj, zz],
                        c_star_temp[:, kk, jj, zz],
                        npar.k,
                        npar.k,
                    )
                    # Check for binding borrowing constraints, no extrapolation from grid
                    bcpol = k_star_temp[1, kk, jj, zz]
                    for mm ∈ 1:n[1]
                        if npar.k[mm] .< bcpol
                            c_star[mm, kk, jj, zz] =
                                inc_lab[mm, kk, jj, zz] .+ inc_LA[mm, kk, jj, zz] .-
                                npar.k[1]
                            k_star[mm, kk, jj, zz] = npar.k[1]
                        end
                        # Cut the policy function at the point where the upper constraint binds
                        if npar.k_max .< k_star[mm, kk, jj, zz]
                            k_star[mm, kk, jj, zz] = npar.k_max
                            c_star[mm, kk, jj, zz] =
                                inc_lab[mm, kk, jj, zz] + inc_LA[mm, kk, jj, zz] -
                                npar.k_max
                        end
                    end
                end
            end
        end
    end
end
