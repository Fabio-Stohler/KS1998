#----------------------------------------------------------------------------
# Basic Functions: Utility, marginal utility and its inverse, 
#                  Return on capital, Wages, Employment, Output
#---------------------------------------------------------------------------

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

# Incomes (K:capital, Z: TFP): Interest rate = MPK.-δ, Wage = MPL, profits = Y-wL-(r+\delta)*K
interest(K, Z, N, mpar::ModelParameters) =
    Z .* mpar.α .* (K ./ N) .^ (mpar.α .- 1.0) .- mpar.δ
wage(K, Z, N, mpar::ModelParameters) =
    Z .* (1 - mpar.α) .* (K ./ N) .^ mpar.α

output(K::Number, Z::Number, N::Number, mpar::ModelParameters) =
    Z .* K .^ (mpar.α) .* N .^ (1 - mpar.α)
