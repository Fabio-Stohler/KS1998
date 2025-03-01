function MakeWeights(xpol, grid)
    idx = zeros(Int64, size(xpol))
    weightright = zeros(typeof(xpol[1]), size(xpol))
    weightleft = zeros(typeof(xpol[1]), size(xpol))
    dx = diff(grid)

    for i in eachindex(xpol)
        if xpol[i] .<= grid[1]
            idx[i] = 1
        elseif xpol[i] .>= grid[end]
            idx[i] = length(grid) .- 1
        else
            idx[i] = locate(xpol[i], grid)
        end
        weightright[i] = (xpol[i] - grid[idx[i]]) / dx[idx[i]]

        weightleft[i] = 1.0 - weightright[i]
    end
    return idx, weightright, weightleft
end

function MakeWeightsLight(xpol, grid)
    idx = zeros(Int64, size(xpol))
    weightright = zeros(eltype(xpol), size(xpol))
    dx = diff(grid)
    @inbounds begin
        for i in eachindex(xpol)
            if xpol[i] .<= grid[1]
                idx[i] = 1
            elseif xpol[i] .>= grid[end]
                idx[i] = length(grid) .- 1
            else
                idx[i] = locate(xpol[i], grid)
            end
            weightright[i] = (xpol[i] .- grid[idx[i]]) ./ dx[idx[i]]
        end
    end
    return idx, weightright
end
