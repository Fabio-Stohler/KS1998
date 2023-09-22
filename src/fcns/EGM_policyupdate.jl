@doc raw"""
    EGM_policyupdate(EVk,r_minus,inc,npar,mpar,warnme)

Find optimal policies, given marginal continuation value `EVk`, today's interest rate `r_minus`, and income `inc`, using the Endogenous Grid Method.

Optimal policies are defined on the fixed grid, but optimal choices (`c` and `k`) are off-grid values.

# Returns
- `c_star`,`k_star`: optimal (on-grid) policies for
    consumption [`c`], and [`k`] asset
"""
function EGM_policyupdate(
    EVk::Array,
    r_minus::Array,
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

function EGM_policyupdate!(
    c_star::Array,
    k_star::Array,
    EMU::Array,
    c_star_temp::Array,
    k_star_temp::Array,
    EVk::Array,
    r_minus::Array,
    inc::Array,
    npar::NumericalParameters,
    mpar::ModelParameters,
    warnme::Bool,
)

    ################### Copy/read-out stuff#####################################
    β::Float64 = mpar.β

    # inc[1] = labor income , 
    # inc[2]= liquid assets income
    inc_lab = inc[1]
    inc_LA = inc[2]
    n = size(EVk)

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EMU .= EVk .* β
    KS.invmutil!(c_star_temp, EMU, mpar) # 6% of time with rolled out power function

    # Calculate assets consistent with choices being [m']
    # Calculate initial money position from the budget constraint
    # that leads to the optimal consumption choice
    k_star_temp .= (c_star_temp .+ npar.mesh_k .- inc_lab)
    # Apply correct interest rate
    k_star_temp .= k_star_temp ./ (r_minus)  # apply borrowing rate

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,m"], m(s,h,m")
    # Interpolate grid().m and c_n_aux defined on m_star_n over grid().m

    # Check monotonicity of m_star_n
    if warnme
        k_star_aux = reshape(k_star_temp, (n[1], n[2], n[3], n[4]))
        if any(any(diff(k_star_aux, dims = 1) .< 0))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies for tuples (c*,m*,y) are now given. Need to interpolate to return to
    # fixed grid.

    @inbounds @views begin
        for jj = 1:n[4] # Loop over income states
            for zz = 1:n[3] # Loop over productivity states
                for kk = 1:n[2] # Loop over aggregate capital states
                    mylinearinterpolate_mult2!(
                        c_star[:, kk, zz, jj],
                        k_star[:, kk, zz, jj],
                        k_star_temp[:, kk, zz, jj],
                        c_star_temp[:, kk, zz, jj],
                        npar.k,
                        npar.k,
                    )
                    # Check for binding borrowing constraints, no extrapolation from grid
                    bcpol = k_star_temp[1, kk, zz, jj]
                    for mm = 1:n[1]
                        if npar.k[mm] .< bcpol
                            c_star[mm, kk, zz, jj] =
                                inc_lab[mm, kk, zz, jj] .+ inc_LA[mm, kk, zz, jj] .-
                                npar.k[1]
                            k_star[mm, kk, zz, jj] = npar.k[1]
                        end
                        if npar.k_max .< k_star[mm, kk, jj]
                            k_star[mm, kk, jj] = npar.k_max
                            c_star[mm, kk, jj] =
                                inc_lab[mm, kk, jj] + inc_LA[mm, kk, jj] - npar.k_max
                        end
                    end
                end
            end
        end
    end
end
