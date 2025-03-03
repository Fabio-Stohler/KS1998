@doc raw"""
    update_EVk(rU::Array, B::Array, mpar::ModelParameters, npar::NumericalParameters)

Returns the expected marginal value of capital tomorrow.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> npar = NumericalParameters()
julia> B = zeros(npar.nstates_ag, npar.nstates_ag)
julia> B[:, 1] = 0.1
julia> B[:, 2] = 0.9
julia> rU = mutil(range(0.1, 5.0, 100))
julia> update_EVk(rU, B, mpar, npar)
```

# Inputs
- `rU::Array`: marginal utility of consumption tomorrow
- `B::Array`: coefficients of the aggregate law of motion
- `mpar::ModelParameters`: model parameters
- `npar::NumericalParameters`: numerical parameters

# Returns
- `EVk::Array`: expected marginal value of capital tomorrow
"""
function update_EVk(
    rU::Array{Float64,4},
    B::Array{Float64,2},
    mpar::ModelParameters,
    npar::NumericalParameters,
)
    # Setup the necessary arrays
    EVk = similar(rU)
    Vk = zeros(npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id, npar.nstates_ag)
    # Update the expected marginal value of capital tomorrow
    update_EVk!(EVk, Vk, rU, B, mpar, npar)
    return EVk
end

@doc raw"""
    update_EVk!(EVk::Array, Vk::Array, rmu::Array, B::Array, mpar::ModelParameters, npar::NumericalParameters)

Returns the expected marginal value of capital tomorrow using inplace operations.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> npar = NumericalParameters()
julia> B = zeros(npar.nstates_ag, npar.nstates_ag)
julia> B[:, 1] = 0.1
julia> B[:, 2] = 0.9
julia> rU = mutil(range(0.1, 5.0, 100))
julia> EVk = similar(rU)
julia> Vk = zeros(npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id, npar.nstates_ag)
julia> update_EVk!(EVk, Vk, rU, B, mpar, npar)
```

# Inputs
- `EVk::Array`: expected marginal value of capital tomorrow
- `Vk::Array`: marginal value of capital tomorrow
- `rmu::Array`: marginal utility of consumption tomorrow
- `B::Array`: coefficients of the aggregate law of motion
- `mpar::ModelParameters`: model parameters
- `npar::NumericalParameters`: numerical parameters
"""
function update_EVk!(
    EVk::Array{Float64,4},
    Vk::Array{Float64,5},
    rmu::Array{Float64,4},
    B::Array{Float64,2},
    mpar::ModelParameters,
    npar::NumericalParameters,
)
    # Export necessary arrays
    km = npar.km
    Π = npar.Π

    # Interpolate on tomorrows capital stock
    km_prime = exp.(B[:, 1] .+ B[:, 2] * log.(km)')
    # Conditional on todays productivity grid ZZ, interpolate onto tomorrows capital grid
    # Dependence follows from the LOM which depends on ZZ
    for yy ∈ 1:(npar.nstates_id) # Tomorrows income grid
        for ZZ ∈ 1:(npar.nstates_ag) # Todays productivity
            # Interpolate on tomorrows capital stock
            Vk[:, :, yy, :, ZZ] .= mylinearinterpolate3(
                npar.k,
                km,
                npar.a,
                rmu[:, :, yy, :],
                npar.k,
                km_prime[ZZ, :],
                npar.a,
            ) # Considerably faster than looping over all states
        end
    end

    # Taking expectations over the marginal value
    for yy ∈ 1:(npar.nstates_id) # Current income state
        for zz ∈ 1:(npar.nstates_ag) # Current aggregate productivity
            # For each (individual) state today there exist four states tomorrow
            EVk[:, :, yy, zz] =
                reshape(
                    Vk[:, :, :, :, zz],
                    (npar.ngridk * npar.ngridkm, npar.nstates_ag * npar.nstates_id),
                ) * Π[(zz .- 1) * 2 + yy, :]
        end
    end
end

# identical function with NumericalParametersDelta

@doc raw"""
    update_EVk(rU::Array, B::Array, mpar::ModelParameters, npar::NumericalParametersDelta)

Returns the expected marginal value of capital tomorrow for the economy where shocks to Delta occure.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> npar = NumericalParametersDelta()
julia> B = zeros(npar.nstates_ag, npar.nstates_ag)
julia> B[:, 1] = 0.1
julia> B[:, 2] = 0.9
julia> rU = mutil(range(0.1, 5.0, 100))
julia> update_EVk(rU, B, mpar, npar)
```

# Inputs

# Inputs
- `EVk::Array`: expected marginal value of capital tomorrow
- `Vk::Array`: marginal value of capital tomorrow
- `rmu::Array`: marginal utility of consumption tomorrow
- `B::Array`: coefficients of the aggregate law of motion
- `mpar::ModelParameters`: model parameters
- `npar::NumericalParametersDelta`: numerical parameters
"""
function update_EVk!(
    EVk::Array{Float64,4},
    Vk::Array{Float64,5},
    rmu::Array{Float64,4},
    B::Array{Float64,2},
    mpar::ModelParameters,
    npar::NumericalParametersDelta,
)
    # Export necessary arrays
    km = npar.km
    Π = npar.Π

    # Interpolate on tomorrows capital stock
    km_prime = exp.(B[:, 1] .+ B[:, 2] * log.(km)')
    # Conditional on todays productivity grid ZZ, interpolate onto tomorrows capital grid
    # Dependence follows from the LOM which depends on ZZ
    for yy ∈ 1:(npar.nstates_id) # Tomorrows income grid
        for ZZ ∈ 1:(npar.nstates_ag) # Todays productivity
            # Interpolate on tomorrows capital stock
            Vk[:, :, yy, :, ZZ] .= mylinearinterpolate3(
                npar.k,
                km,
                npar.δ,
                rmu[:, :, yy, :],
                npar.k,
                km_prime[ZZ, :],
                npar.δ,
            ) # Considerably faster than looping over all states
        end
    end

    # Taking expectations over the marginal value
    for yy ∈ 1:(npar.nstates_id) # Current income state
        for zz ∈ 1:(npar.nstates_ag) # Current aggregate productivity
            # For each (individual) state today there exist four states tomorrow
            EVk[:, :, yy, zz] =
                reshape(
                    Vk[:, :, :, :, zz],
                    (npar.ngridk * npar.ngridkm, npar.nstates_ag * npar.nstates_id),
                ) * Π[(zz .- 1) * 2 + yy, :]
        end
    end
end

@doc raw"""
    update_EVk(rU::Array, B::Array, mpar::ModelParameters, npar::NumericalParametersDelta)

Returns the expected marginal value of capital tomorrow for the economy where shocks to the discount factor occure.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> npar = NumericalParametersDelta()
julia> B = zeros(npar.nstates_ag, npar.nstates_ag)
julia> B[:, 1] = 0.1
julia> B[:, 2] = 0.9
julia> rU = mutil(range(0.1, 5.0, 100))
julia> update_EVk(rU, B, mpar, npar)
```

# Inputs

# Inputs
- `EVk::Array`: expected marginal value of capital tomorrow
- `Vk::Array`: marginal value of capital tomorrow
- `rmu::Array`: marginal utility of consumption tomorrow
- `B::Array`: coefficients of the aggregate law of motion
- `mpar::ModelParameters`: model parameters
- `npar::NumericalParametersDelta`: numerical parameters
"""
function update_EVk!(
    EVk::Array{Float64,4},
    Vk::Array{Float64,5},
    rmu::Array{Float64,4},
    B::Array{Float64,2},
    mpar::ModelParameters,
    npar::NumericalParametersBeta,
)
    # Export necessary arrays
    km = npar.km
    Π = npar.Π

    # Interpolate on tomorrows capital stock
    km_prime = exp.(B[:, 1] .+ B[:, 2] * log.(km)')
    # Conditional on todays productivity grid ZZ, interpolate onto tomorrows capital grid
    # Dependence follows from the LOM which depends on ZZ
    for yy ∈ 1:(npar.nstates_id) # Tomorrows income grid
        for ZZ ∈ 1:(npar.nstates_ag) # Todays productivity
            # Interpolate on tomorrows capital stock
            Vk[:, :, yy, :, ZZ] .= mylinearinterpolate3(
                npar.k,
                km,
                npar.β,
                rmu[:, :, yy, :],
                npar.k,
                km_prime[ZZ, :],
                npar.β,
            ) # Considerably faster than looping over all states
        end
    end

    # Taking expectations over the marginal value
    for yy ∈ 1:(npar.nstates_id) # Current income state
        for zz ∈ 1:(npar.nstates_ag) # Current aggregate productivity
            # For each (individual) state today there exist four states tomorrow
            EVk[:, :, yy, zz] =
                reshape(
                    npar.mesh_β ./ npar.β[zz] .* Vk[:, :, :, :, zz], # accounting for the discount factor shock
                    (npar.ngridk * npar.ngridkm, npar.nstates_ag * npar.nstates_id),
                ) * Π[(zz .- 1) * 2 + yy, :]
        end
    end
end

@doc raw"""
    update_EVk(rU::Array, B::Array, mpar::ModelParameters, npar::NumericalParametersAll)

Returns the expected marginal value of capital tomorrow for the economy where shocks to the discount factor occure.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> npar = NumericalParametersDelta()
julia> B = zeros(npar.nstates_ag, npar.nstates_ag)
julia> B[:, 1] = 0.1
julia> B[:, 2] = 0.9
julia> rU = mutil(range(0.1, 5.0, 100))
julia> update_EVk(rU, B, mpar, npar)
```

# Inputs

# Inputs
- `EVk::Array`: expected marginal value of capital tomorrow
- `Vk::Array`: marginal value of capital tomorrow
- `rmu::Array`: marginal utility of consumption tomorrow
- `B::Array`: coefficients of the aggregate law of motion
- `mpar::ModelParameters`: model parameters
- `npar::NumericalParametersDelta`: numerical parameters
"""
function update_EVk!(
    EVk::Array{Float64,4},
    Vk::Array{Float64,5},
    rmu::Array{Float64,4},
    B::Array{Float64,2},
    mpar::ModelParameters,
    npar::NumericalParametersAll,
)
    # Export necessary arrays
    km = npar.km
    Π = npar.Π

    # Interpolate on tomorrows capital stock
    km_prime = exp.(B[:, 1] .+ B[:, 2] * log.(km)')
    # Conditional on todays productivity grid ZZ, interpolate onto tomorrows capital grid
    # Dependence follows from the LOM which depends on ZZ
    for yy ∈ 1:(npar.nstates_id) # Tomorrows income grid
        for ZZ ∈ 1:(npar.nstates_ag) # Todays productivity
            # Interpolate on tomorrows capital stock
            Vk[:, :, yy, :, ZZ] .= mylinearinterpolate3(
                npar.k,
                km,
                Array(1:8),
                rmu[:, :, yy, :],
                npar.k,
                km_prime[ZZ, :],
                Array(1:8),
            ) # Considerably faster than looping over all states
        end
    end

    # Taking expectations over the marginal value
    for yy ∈ 1:(npar.nstates_id) # Current income state
        for zz ∈ 1:(npar.nstates_ag) # Current aggregate productivity
            # For each (individual) state today there exist four states tomorrow
            EVk[:, :, yy, zz] =
                reshape(
                    npar.mesh_β ./ npar.mesh_β[1, 1, 1, zz] .* Vk[:, :, :, :, zz], # accounting for the discount factor shock
                    (npar.ngridk * npar.ngridkm, npar.nstates_ag * npar.nstates_id),
                ) * Π[(zz .- 1) * 2 + yy, :]
        end
    end
end
