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
    EVk::Array{Float64, 4},
    r_minus::Array{Float64, 4},
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
    c_star::Array{Float64, 4},
    k_star::Array{Float64, 4},
    EMU::Array{Float64, 4},
    c_star_temp::Array{Float64, 4},
    k_star_temp::Array{Float64, 4},
    EVk::Array{Float64, 4},
    r_minus::Array{Float64, 4},
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
        if any(any(diff(k_star_aux, dims = 1) .< 0))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies on grid tuples (k, km, z, e) are now given. 
    # Need to interpolate to return to fixed grid.
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
                        # Cut the policy function at the point where the upper constraint binds
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


@doc raw"""
    EGM_policyupdate(EVm, EVk, Qminus, Tshock, inc, npar, mpar, warnme)

Find optimal policies, given marginal continuation values `EVb`, `EVk`, and income [`inc`], using the Endogenous Grid Method.

Optimal policies are defined on the fixed grid, but optimal asset choices (`b` and `k`) are off-grid values.

# Returns
- `c_star`, `b_star`, and `k_star`: optimal (on-grid) policies for
    consumption [`c`], bonds [`b`] and capital [`k`] asset
"""
function EGM_policyupdate(
    EVb::Array,
    EVk::Array,
    inc::Array,
    Qminus::Array,
    Tshock::Float64,
    npar::NumericalParameters,
    mpar::ModelParameters,
    warnme::Bool,
)
    # Pre-Allocate returns
    c_star = similar(EVb) # Initialize c_n-container
    b_star = similar(EVb) # Initialize m_n-container
    k_star = similar(EVb) # Initialize k_a-container
    # containers for auxiliary variables
    E_return_diff = similar(EVb)
    EMU = similar(EVb)
    c_star_temp = similar(EVb)
    b_temp = similar(EVb)
    c_star_single = similar(EVb)
    b_star_single = similar(EVb)
    n = size(EVb)
    Resource_grid = reshape(inc[2] .+ inc[3], (n[1] .* n[2], n[3], n[4], n[5]))
    EGM_policyupdate!(
        c_star,
        b_star,
        k_star,
        E_return_diff,
        EMU,
        c_star_temp,
        b_temp,
        c_star_single,
        b_star_single,
        Resource_grid,
        EVb,
        EVk,
        Qminus,
        Tshock,
        inc,
        npar,
        mpar,
        warnme,
    )
    return c_star, b_star, k_star, c_star_temp, c_star_single, b_temp
end

function EGM_policyupdate!(
    c_star::Array,
    b_star::Array,
    k_star::Array,
    E_return_diff::Array,
    EMU::Array,
    c_star_temp::Array,
    b_temp::Array,
    c_star_single::Array,
    b_star_single::Array,
    Resource_grid::Array,
    EVb::Array,
    EVk::Array,
    Qminus::Array,
    Tshock::Float64,
    inc::Array,
    npar::NumericalParameters,
    mpar::ModelParameters,
    warnme::Bool,
)
    ################### Copy/read-out stuff#####################################
    # Test EGM function here
    β::Float64 = mpar.β
    borrwedge = mpar.Rbar .* Tshock
    # inc[1] = labor income , inc[2] = capital income,
    # inc[3]= bond income 
    inc_lab = inc[1]
    inc_bond = inc[2]
    inc_cap = inc[3]
    n = size(EVb)

    ############################################################################
    ## EGM Step 1: Find optimal liquid asset holdings in the constrained case ##
    ############################################################################
    EMU .= EVb .* β ./ Qminus
    KS.invmutil!(c_star_temp, EMU, mpar) # 6% of time with rolled out power function

    # Next step: Interpolate w_guess and c_guess from new k-grids
    # using c[s,h,b"], b(s,h,b")
    # Interpolate grid().b and c_aux defined on b_temp over grid().b

    # Check monotonicity of b_temp
    if warnme
        b_star_aux = reshape(b_temp, (n[1], n[2] * n[3] * n[4] * n[5]))
        if any(any(diff(b_star_aux, dims = 1) .< 0))
            @warn "non monotone future liquid asset choice encountered"
        end
    end

    # Policies for tuples (c*,b*,y) are now given. Need to interpolate to return to
    # fixed grid.
    @inbounds begin
        for jj = 1:n[5] # Loop over employment states
            for zz = 1:n[4] # Loop over productivity states
                for KK = 1:n[3] # Loop over aggregate capital
                    for kk = 1:n[2] # Loop over capital states
                        c_star_single[:, kk, KK, zz, jj],b_star_single[:, kk, KK, zz, jj] = KS.mylinearinterpolate_mult2(
                            b_temp[:, kk, KK, zz, jj],
                            c_star_temp[:, kk, KK, zz, jj],
                            npar.b,
                            npar.b,
                        )
                        # KS.mylinearinterpolate_mult2!(
                            # c_star_single[:, kk, KK, zz, jj],
                            # b_star_single[:, kk, KK, zz, jj],
                            # b_temp[:, kk, KK, zz, jj],
                            # c_star_temp[:, kk, KK, zz, jj],
                            # npar.b,
                            # npar.b,
                        # )
                        # Check for binding borrowing constraints, no extrapolation from grid
                        bcpol = b_temp[1, kk, KK, zz, jj]
                        for mm = 1:n[1]
                            if npar.b[mm] .< bcpol
                                c_star_single[mm, kk, KK, zz, jj] = inc_lab[mm, kk, KK, zz, jj] .+ (inc_cap[mm, kk, KK, zz, jj] .- npar.mesh_k[mm, kk, KK, zz, jj]) .+ inc_bond[mm, kk, KK, zz, jj] .- Qminus[mm, kk, KK, zz, jj] .* npar.b[1]
                                b_star_single[mm, kk, KK, zz, jj] = npar.b[1]
                            end
                            if npar.b_max .< b_star_single[mm,kk,KK,zz,jj]
                                b_star_single[mm,kk,KK,zz,jj] = npar.b_max
                            end
                        end
                    end
                end
            end
        end
    end

    
    #-------------------------END OF STEP 1-----------------------------

    ############################################################################
    ## EGM Step 2: Find Optimal Portfolio Combinations                        ##
    ############################################################################


    # Find an b* for given k' that yield the same expected future marginal value
    # for bonds and capital:
    E_return_diff .= β .* EVk .- EMU           # difference conditional on future asset holdings on grid
    b_aux1 = KS.Fastroot(npar.b, -E_return_diff)  # Find indifferent b by interpolation of two neighboring points a, b ∈ grid_m with:  E_return_diff(a) < 0 < E_return_diff(b)
    # (Fastroot does not allow for extrapolation and uses non-negativity constraint and monotonicity)
    b_aux = reshape(b_aux1, (n[2], n[3], n[4], n[5]))

    ###########################################################################
    ## EGM Step 3: Constraints for bonds and capital are not binding         ##
    ###########################################################################
    # Interpolation of psi()-function at b*[b,k]
    aux_index = (0:(n[2]*n[3]*n[4]*n[5])-1) * n[1]                        # auxiliary to move to linear indexing
    EMU_star = Array{eltype(b_aux), 4}(undef, (n[2], n[3], n[4], n[5]))   # container
    steps = diff(npar.b)                            # Stepsize on grid()

    # Interpolate EMU[b",k',s'*h',M',K'] over b*_n[k"], b-dim is dropped
    for j in eachindex(b_aux)
        xi = b_aux[j]
        # find indexes on grid next smallest to optimal policy
        if xi .> npar.b[n[1]-1]                                # policy is larger than highest grid point 
            idx = n[1] - 1
        elseif xi .<= npar.b[1]                                # policy is smaller than lowest grid point
            idx = 1
        else
            idx = locate(xi, npar.b)                       # use exponential search to find grid point closest to policy (next smallest)
        end

        s = (xi .- npar.b[idx]) ./ steps[idx]          # Distance of optimal policy to next grid point

        EMU_star[j] =
            EMU[idx.+aux_index[j]] .* (1.0 - s) .+ s .* (EMU[idx.+aux_index[j].+1])        # linear interpolation
    end

    # Calculate optimal consumption choice
    c_aux = KS.invmutil(EMU_star, mpar)

    # Resources that lead to capital choice
    # k'= c + b*(k") + k" - w*h
    # = value of todays cap and bond holdings
    Resource = c_aux .+ Qminus[1, :, :, :, :] .* b_aux .+ npar.mesh_k[1, :, :, :, :] .- inc_lab[1, :, :, :, :]

    # Bond constraint is not binding, but capital constraint is binding
    b_star_zero = b_aux[1, :, :, :] # bond holdings that correspond to k'=0:  b*(k=0)

    # Use consumption at k"=0 from constrained problem, when b" is on grid()
    aux_c = reshape(c_star_temp[:, 1, :, :, :], (n[1], n[3], n[4], n[5]))
    aux_inc = reshape(inc_lab[1, 1, :, :, :], (n[3], n[4], n[5]))
    cons_list = Array{Array{eltype(c_star_single)}}(undef, n[3], n[4], n[5])
    res_list = Array{Array{eltype(c_star_single)}}(undef, n[3], n[4], n[5])
    bond_list = Array{Array{eltype(c_star_single)}}(undef, n[3], n[4], n[5])
    cap_list = Array{Array{eltype(c_star_single)}}(undef, n[3], n[4], n[5])
    log_index = Vector{Bool}(undef, npar.ngridb)
    for j = 1:n[5]
        for z = 1:n[4]
            for km = 1:n[3]
                # When choosing zero capital holdings, HHs might still want to choose bond 
                # holdings smaller than b*(k'=0)
                if b_star_zero[km, z, j] > npar.b[1]
                    # Calculate consumption policies, when HHs chooses bond holdings
                    # lower than b*(k"=0) and capital holdings k"=0 and save them in cons_list
                    log_index .= npar.b .< b_star_zero[km, z, j]
                    # aux_c is the consumption policy under no capital (fix k=0), for b<b*(k'=0)
                    c_k_cons = aux_c[log_index, km, z, j]
                    cons_list[km, z, j] = c_k_cons  #Consumption at k"=0, b"<m*(0)
                    # Required Resources: Bond choice + Consumption - labor income
                    # Resources that lead to k"=0 and b'<b*(k"=0)
                    bonds = npar.b[log_index]
                    bond_list[km, z, j] = bonds
                    res_list[km, z, j] = Qminus[log_index, 1, km, z, j] .* bonds .+ c_k_cons .- aux_inc[km, z, j]
                    cap_list[km, z, j] = zeros(eltype(EVb), sum(log_index))
                else
                    cons_list[km, z, j] = zeros(eltype(EVb), 0) #Consumption at k"=0, b"<b*(0)
                    # Required Resources: Bond choice + Consumption - labor income
                    # Resources that lead to k"=0 and b'<b*(k"=0)
                    res_list[km, z, j] = zeros(eltype(EVb), 0)
                    bond_list[km, z, j] = zeros(eltype(EVb), 0)
                    cap_list[km, z, j] = zeros(eltype(EVb), 0)
                end
            end
        end
    end

    # Merge lists
    c_aux = reshape(c_aux, (n[2], n[3], n[4], n[5]))
    b_aux = reshape(b_aux, (n[2], n[3], n[4], n[5]))

    for j = 1:n[5]
        for z = 1:n[4]
            for km = 1:n[3]
                append!(cons_list[km, z, j], c_aux[:, km, z, j])
                append!(res_list[km, z, j], Resource[:, km, z, j])
                append!(bond_list[km, z, j], b_aux[:, km, z, j])
                append!(cap_list[km, z, j], npar.k)
            end
        end
    end

    ####################################################################
    ## EGM Step 4: Interpolate back to fixed grid                     ##
    ####################################################################
    labor_inc_grid = inc_lab[1, 1, :, :, :]
    log_index2 = zeros(Bool, n[1] .* n[2])

    @views @inbounds begin
        for j = 1:n[5]
            for z = 1:n[4]
                for km = 1:n[3]
                    # Check monotonicity of resources
                    if warnme
                        if any(diff(res_list[km, z, j]) .< 0)
                            @warn "non monotone resource list encountered"
                        end
                    end

                    # when at most one constraint binds:
                    # Lowest value of res_list corresponds to m_a"=0 and k_a"=0.
                    KS.mylinearinterpolate_mult3!(
                        c_star[:, :, km, z, j][:],
                        b_star[:, :, km, z, j][:],
                        k_star[:, :, km, z, j][:],
                        res_list[km, z, j],
                        cons_list[km, z, j],
                        bond_list[km, z, j],
                        cap_list[km, z, j],
                        Resource_grid[:, km, z, j],
                    )

                    # Any resources on grid smaller then res_list imply that HHs consume all
                    # resources plus income.
                    # When both constraints are binding:
                    log_index2[:] .= reshape(Resource_grid[:, km, z, j], n[1] * n[2]) .< res_list[km, z, j][1]
                    c_star[:, :, km, z, j][log_index2] .=
                        Resource_grid[log_index2, km, z, j] .+ labor_inc_grid[km, z, j] .- Qminus[:, :, km, z, j][log_index2] .* npar.b[1]
                    b_star[:, :, km, z, j][log_index2] .= npar.b[1]
                    k_star[:, :, km, z, j][log_index2] .= 0.0
                end
            end
        end
    end
end