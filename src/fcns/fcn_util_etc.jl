# Basic Functions: Utility, marginal utility and its inverse, Return on capital, Wages, Output

@doc raw"""
    util(c::AbstractArray, mpar::ModelParameters)

Returns utility from consumption `c`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> c = range(0.1, 1.0, length = 10)
julia> util(c, mpar)

# Inputs
- `c::AbstractArray`: consumption Array
- `mpar::ModelParameters`: model parameters

# Returns
- `util::AbstractArray`: utility from consumption
"""
function util(c::AbstractArray, mpar::ModelParameters)
    if mpar.σ == 1.0
        util = log.(c)
    elseif mpar.σ == 2.0
        util = 1.0 - 1.0 ./ c
    elseif mpar.σ == 4.0
        util = (1.0 - 1.0 ./ (c .* c .* c)) ./ 3.0
    else
        util = (c .^ (1.0 .- mpar.σ) .- 1.0) ./ (1.0 .- mpar.σ)
    end
    return util
end

@doc raw"""
    mutil(c::AbstractArray, mpar::ModelParameters)

Returns marginal utility from consumption `c`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> c = range(0.1, 1.0, length = 10)
julia> mutil(c, mpar)

# Inputs
- `c::AbstractArray`: consumption Array
- `mpar::ModelParameters`: model parameters

# Returns
- `mutil::AbstractArray`: marginal utility from consumption
"""
function mutil(c::AbstractArray, mpar::ModelParameters)
    if mpar.σ == 1.0
        mutil = 1.0 ./ c
    elseif mpar.σ == 2.0
        mutil = 1.0 ./ (c .^ 2)
    elseif mpar.σ == 4.0
        mutil = 1.0 ./ ((c .^ 2) .^ 2)
    else
        mutil = c .^ (-mpar.σ)
    end
    return mutil
end

@doc raw"""
    mutil!(mu::AbstractArray, c::AbstractArray, mpar::ModelParameters)

Returns marginal utility from consumption `c` and stores it in `mu`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> c = range(0.1, 1.0, length = 10)
julia> mu = similar(c)
julia> mutil!(mu, c, mpar)

# Inputs
- `mu::AbstractArray`: marginal utility from consumption
- `c::AbstractArray`: consumption Array
- `mpar::ModelParameters`: model parameters
"""
function mutil!(mu::AbstractArray, c::AbstractArray, mpar::ModelParameters)
    if mpar.σ == 1.0
        mu .= 1.0 ./ c
    elseif mpar.σ == 2.0
        mu .= 1.0 ./ (c .^ 2)
    elseif mpar.σ == 4.0
        mu .= 1.0 ./ ((c .^ 2) .^ 2)
    else
        mu .= c .^ (-mpar.σ)
    end
    return mu
end

@doc raw"""
    invmutil(mu::AbstractArray, mpar::ModelParameters)

Returns inverse marginal utility from consumption `mu`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> mu = range(0.1, 1.0, length = 10)
julia> invmutil(mu, mpar)

# Inputs
- `mu::AbstractArray`: marginal utility from consumption
- `mpar::ModelParameters`: model parameters

# Returns
- `c::AbstractArray`: inverse marginal utility
"""
function invmutil(mu, mpar::ModelParameters)
    if mpar.σ == 1.0
        c = 1.0 ./ mu
    elseif mpar.σ == 2.0
        c = 1.0 ./ (sqrt.(mu))
    elseif mpar.σ == 4.0
        c = 1.0 ./ (sqrt.(sqrt.(mu)))
    else
        c = 1.0 ./ mu .^ (1.0 ./ mpar.σ)
    end
    return c
end

@doc raw"""
    invmutil!(c::AbstractArray, mu::AbstractArray, mpar::ModelParameters)

Returns inverse marginal utility from consumption `mu` and stores it in `c`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> mu = range(0.1, 1.0, length = 10)
julia> c = similar(mu)
julia> invmutil!(c, mu, mpar)

# Inputs
- `c::AbstractArray`: inverse marginal utility
- `mu::AbstractArray`: marginal utility from consumption
- `mpar::ModelParameters`: model parameters
"""
function invmutil!(c, mu, mpar::ModelParameters)
    if mpar.σ == 1.0
        c .= 1.0 ./ mu
    elseif mpar.σ == 2.0
        c .= 1.0 ./ (sqrt.(mu))
    elseif mpar.σ == 4.0
        c .= 1.0 ./ (sqrt.(sqrt.(mu)))
    else
        c .= 1.0 ./ mu .^ (1.0 ./ mpar.σ)
    end
    return c
end

@doc raw"""
    interest(K::AbstractArray, Z::AbstractArray, N::AbstractArray, mpar::ModelParameters)

Returns interest rate from capital `K`, productivity `Z` and labor `N`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> K = range(0.1, 1.0, length = 10)
julia> Z = range(0.1, 1.0, length = 10)
julia> N = range(0.1, 1.0, length = 10)
julia> interest(K, Z, N, mpar)

# Inputs
- `K::AbstractArray`: capital
- `Z::AbstractArray`: productivity
- `N::AbstractArray`: labor
- `mpar::ModelParameters`: model parameters

# Returns
- `r::AbstractArray`: interest rate
"""
interest(K::Array, Z::Array, N::Array, mpar::ModelParameters) =
    Z .* mpar.α .* (K ./ N) .^ (mpar.α .- 1.0) .- mpar.δ

@doc raw"""
    interest(K::AbstractArray, Z::Float64, N::AbstractArray, δ::AbstractArray, mpar::ModelParameters)

Returns interest rate from capital `K`, productivity `Z`, labor `N` and depreciation `δ`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> K = range(0.1, 1.0, length = 10)
julia> Z = 0.5
julia> N = range(0.1, 1.0, length = 10)
julia> δ = range(0.1, 1.0, length = 10)
julia> interest(K, Z, N, δ, mpar)

# Inputs
- `K::AbstractArray`: capital
- `Z::Float`: productivity
- `N::AbstractArray`: labor
- `δ::AbstractArray`: depreciation
- `mpar::ModelParameters`: model parameters

# Returns
- `r::AbstractArray`: interest rate
"""
interest(K::Array, Z::Float64, N::Array, δ::Array, mpar::ModelParameters) =
    Z .* mpar.α .* (K ./ N) .^ (mpar.α .- 1.0) .- δ

@doc raw"""
    interest(K::AbstractArray, Z::Float64, N::AbstractArray, δ::AbstractArray, mpar::ModelParameters)

Returns interest rate from capital `K`, productivity `Z`, labor `N` and depreciation `δ`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> K = range(0.1, 1.0, length = 10)
julia> Z = 0.5
julia> N = range(0.1, 1.0, length = 10)
julia> δ = range(0.1, 1.0, length = 10)
julia> interest(K, Z, N, δ, mpar)

# Inputs
- `K::AbstractArray`: capital
- `Z::AbstractArray`: productivity
- `N::AbstractArray`: labor
- `δ::AbstractArray`: depreciation
- `mpar::ModelParameters`: model parameters

# Returns
- `r::AbstractArray`: interest rate
"""
interest(K::Array, Z::Array, N::Array, δ::Array, mpar::ModelParameters) =
    Z .* mpar.α .* (K ./ N) .^ (mpar.α .- 1.0) .- δ

@doc raw"""
    wage(K::AbstractArray, Z::AbstractArray, N::AbstractArray, mpar::ModelParameters)

Returns wage from capital `K`, productivity `Z` and labor `N`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> K = range(0.1, 1.0, length = 10)
julia> Z = range(0.1, 1.0, length = 10)
julia> N = range(0.1, 1.0, length = 10)
julia> wage(K, Z, N, mpar)

# Inputs
- `K::AbstractArray`: capital
- `Z::AbstractArray`: productivity
- `N::AbstractArray`: labor
- `mpar::ModelParameters`: model parameters

# Returns
- `w::AbstractArray`: wage
"""
wage(K::Array, Z::Array, N::Array, mpar::ModelParameters) =
    Z .* (1 - mpar.α) .* (K ./ N) .^ mpar.α

@doc raw"""
    output(K::AbstractArray, Z::AbstractArray, N::AbstractArray, mpar::ModelParameters)

Returns output from capital `K`, productivity `Z` and labor `N`.

# Example
```julia-repl
julia> using KS
julia> mpar = ModelParameters()
julia> K = range(0.1, 1.0, length = 10)
julia> Z = range(0.1, 1.0, length = 10)
julia> N = range(0.1, 1.0, length = 10)
julia> output(K, Z, N, mpar)

# Inputs
- `K::AbstractArray`: capital
- `Z::AbstractArray`: productivity
- `N::AbstractArray`: labor
- `mpar::ModelParameters`: model parameters

# Returns
- `Y::AbstractArray`: output
"""
output(K::Array, Z::Array, N::Array, mpar::ModelParameters) =
    Z .* K .^ (mpar.α) .* N .^ (1 - mpar.α)
