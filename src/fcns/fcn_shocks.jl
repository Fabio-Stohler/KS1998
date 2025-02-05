@doc raw"""
    shocks_parameters()

Wrapper function that generates the transition matrix for the aggregate shocks depending on the aggregate shock that hits the economy.

# Returns
- `ur_b::Float64`: unemployment rate in a bad aggregate state
- `er_b::Float64`: employment rate in a bad aggregate state
- `ur_g::Float64`: unemployment rate in a good aggregate state
- `er_g::Float64`: employment rate in a good aggregate state
- `Π::Array`: transition matrix conditional on the aggregate state
- `Π_ag::Array`: transition matrix for the aggregate state
"""
function shocks_parameters(shocks = "a")

    # Check which type of shock is hitting the economy
    if string(shocks) == "a"
        return shocks_parameters_technology()
    elseif string(shocks) == "all"
        return shocks_parameters_all()
    elseif string(shocks) == "β"
        return shocks_parameters_beta_depreciation()
    elseif string(shocks) == "δ"
        return shocks_parameters_beta_depreciation()
    else
        @error "Shocks not found"
    end
end

@doc raw"""
    shocks_parameters_technology()

Function that generates the transition matrix for the aggregate shocks and the transition matrix conditional on the aggregate technology state

# Returns
- `ur_b::Float64`: unemployment rate in a bad aggregate state
- `er_b::Float64`: employment rate in a bad aggregate state
- `ur_g::Float64`: unemployment rate in a good aggregate state
- `er_g::Float64`: employment rate in a good aggregate state
- `Π::Array`: transition matrix conditional on the aggregate state
- `Π_ag::Array`: transition matrix for the aggregate state
"""
function shocks_parameters_technology()
    # Part that has to be sourced out into a function
    # Assumptions on the risk behavior
    D_g = 8                 # Duration of a good aggregate state
    D_b = 8                 # Duration of a bad aggregate state
    ur_b = 0.1              # unemployment rate in a bad aggregate state
    er_b = (1 - ur_b)       # employment rate in a bad aggregate state
    ur_g = 0.04             # unemployment rate in a good aggregate state
    er_g = (1 - ur_g)       # employment rate in a good aggregate state

    # Time to find a job in the respective states
    # Coding is D_ss'ee' where e is the employment state and s is the aggregate state
    D_bb01 = 2.5
    D_gg01 = 1.5

    # Producing the actual transition matrix
    π_gg = 1 - 1 / D_g
    π_bb = 1 - 1 / D_b
    Π_ag = [π_bb 1-π_bb; 1-π_gg π_gg]

    # Transition conditional on the current state
    π_01_bb = 1 / D_bb01
    π_01_gg = 1 / D_gg01

    # Setting up the transition matrix conditional on the states
    Π = zeros((4, 4))
    Π[1, 2] = π_01_bb * π_bb     # Prob(ϵ' = 1, z' = 0| ϵ = 0, z = 0) = Prob(ϵ' = 1| ϵ = 0, z' = 0, z = 0) * Prob(z' = 0| z = 0)
    Π[1, 1] = π_bb - Π[1, 2]      # Π_bb00 + Π_bb01 = Π_bb
    Π[3, 4] = π_01_gg * π_gg     # Prob(ϵ' = 1, z' = 1| ϵ = 0, z = 1) = Prob(ϵ' = 1| ϵ = 0, z' = 1, z = 1) * Prob(z' = 1| z = 1)
    Π[3, 3] = π_gg - Π[3, 4]      # Π_gg00 + Π_gg01 = Π_gg

    # Conditions for transtion Π_gb00 / Π_gb = 1.25 Π_bb00 / Π_bb
    Π[3, 1] = 1.25 * Π[1, 1] / Π_ag[1, 1] * Π_ag[2, 1]
    Π[3, 2] = Π_ag[2, 1] - Π[3, 1] # Π_gb00 + Π_gb01 = Π_gb

    # Conditions for transition Π_bg00 / Π_bg = 0.75 Π_gg00 / Π_gg
    Π[1, 3] = 0.75 * Π[3, 3] / Π_ag[2, 2] * Π_ag[1, 2]
    Π[1, 4] = Π_ag[1, 2] - Π[1, 3] # Π_bg00 + Π_bg01 = Π_bg

    # Imposing the law of motion for unemployment
    # u_s * Π_ss'00 / Π_ss' + (1 - u_s) * Π_ss'10 / Π_ss' = u_s'
    Π[2, 1] = ur_b * π_bb / (1 - ur_b) * (1 - Π[1, 1] / π_bb)
    Π[2, 2] = π_bb - Π[2, 1] # Π_bb10 + Π_bb11 = Π_bb
    Π[2, 3] = Π_ag[1, 2] / (1 - ur_b) * (ur_g - ur_b * Π[1, 3] / Π_ag[1, 2])
    Π[2, 4] = Π_ag[1, 2] - Π[2, 3] # Π_bg10 + Π_bg11 = Π_bg
    Π[4, 1] = Π_ag[2, 1] / (1 - ur_g) * (ur_b - ur_g * Π[3, 1] / Π_ag[2, 1])
    Π[4, 2] = Π_ag[2, 1] - Π[4, 1] # Π_gb10 + Π_gb11 = Π_gb
    Π[4, 3] = ur_g * π_gg / (1 - ur_g) * (1 - Π[3, 3] / π_gg)
    Π[4, 4] = π_gg - Π[4, 3] # Π_gg10 + Π_gg11 = Π_gg

    return ur_b, er_b, ur_g, er_g, Π, Π_ag
end

@doc raw"""
    shocks_parameters_beta_depreciation()

Function that generates the transition matrix for the aggregate shocks and the transition matrix conditional on the aggregate depreciation state

# Returns
- `ur_b::Float64`: unemployment rate in a bad aggregate state
- `er_b::Float64`: employment rate in a bad aggregate state
- `ur_g::Float64`: unemployment rate in a good aggregate state
- `er_g::Float64`: employment rate in a good aggregate state
- `Π::Array`: transition matrix conditional on the aggregate state
- `Π_ag::Array`: transition matrix for the aggregate state
"""

function shocks_parameters_beta_depreciation()
    # Part that has to be sourced out into a function
    # Assumptions on the risk behavior
    D_g = 8                 # Duration of a good aggregate state
    D_b = 2                 # Duration of a bad aggregate state
    ur_b = 0.1              # unemployment rate in a bad aggregate state
    er_b = (1 - ur_b)       # employment rate in a bad aggregate state
    ur_g = 0.04             # unemployment rate in a good aggregate state
    er_g = (1 - ur_g)       # employment rate in a good aggregate state

    # Time to find a job in the respective states
    # Coding is D_ss'ee' where e is the employment state and s is the aggregate state
    D_bb01 = 2.5
    D_gg01 = 1.5

    # Producing the actual transition matrix
    π_gg = 1 - 1 / D_g
    π_bb = 1 - 1 / D_b
    Π_ag = [π_bb 1-π_bb; 1-π_gg π_gg]

    # Transition conditional on the current state
    π_01_bb = 1 / D_bb01
    π_01_gg = 1 / D_gg01

    # Setting up the idiosyncratic transition matrix
    Π_id = zeros(2, 2)

    # Use duration in unemployment to calculate the transition matrix
    Π_id[1, 2] = π_01_gg                # Prob(ϵ' = 1| ϵ = 0) = Prob(ϵ' = 1| ϵ = 0)
    Π_id[1, 1] = 1 - Π_id[1, 2]         # Π_00 + Π_01 = 1

    # Use the law of motion of unemployment
    Π_id[2, 1] = (1 - Π_id[1, 1]) * ur_g / (1 - ur_g) # unemployment transition from good to good
    Π_id[2, 2] = 1 - Π_id[2, 1]          # Π_10 + Π_11 = 1

    # create aggregate transition matrix via kronecker product
    Π = kron(Π_ag, Π_id)

    return ur_b, er_b, ur_g, er_g, Π, Π_ag
end

@doc raw"""
    shocks_parameters_all()

Function that generates the transition matrix for the aggregate shocks and the transition matrix conditional on the aggregate state

# Returns
- `ur_b::Float64`: unemployment rate in a bad aggregate state
- `er_b::Float64`: employment rate in a bad aggregate state
- `ur_g::Float64`: unemployment rate in a good aggregate state
- `er_g::Float64`: employment rate in a good aggregate state
- `Π::Array`: transition matrix conditional on the aggregate state
- `Π_ag::Array`: transition matrix for the aggregate state
"""
function shocks_parameters_all()
    # Part that has to be sourced out into a function
    Da_g = 8                 # Duration of a good techn. state
    Da_b = 8                 # Duration of a bad techn. state
    Dβ_g = 8                 # Duration of a good β state
    Dβ_b = 8                 # Duration of a bad β state
    Dδ_g = 8                 # Duration of a good depreciation state
    Dδ_b = 2                 # Duration of a bad depreciation state

    # Employment and unemployment
    ur_b = 0.1              # unemployment rate in a bad techn. state
    er_b = (1 - ur_b)       # employment rate in a bad techn. state
    ur_g = 0.04             # unemployment rate in a good techn. state
    er_g = (1 - ur_g)       # employment rate in a good techn. state

    # Time to find a job in the respective states
    # Coding is D_ss'ee' where e is the employment state and s is the aggregate state
    D_bb01 = 2.5
    D_gg01 = 1.5

    # Transition matrix for technology
    πa_gg = 1 - 1 / Da_g
    πa_bb = 1 - 1 / Da_b
    Πa_ag = [πa_bb 1-πa_bb; 1-πa_gg πa_gg]

    # Transition matrix for depreciation
    πδ_gg = 1 - 1 / Dδ_g
    πδ_bb = 1 - 1 / Dδ_b
    Πδ_ag = [πδ_bb 1-πδ_bb; 1-πδ_gg πδ_gg]

    # Transition matrix for β
    πβ_gg = 1 - 1 / Dβ_g
    πβ_bb = 1 - 1 / Dβ_b
    Πβ_ag = [πβ_bb 1-πβ_bb; 1-πβ_gg πβ_gg]

    # Transition conditional on the current state
    π_01_bb = 1 / D_bb01
    π_01_gg = 1 / D_gg01

    # Setting up the transition matrix conditional on the states
    Π = zeros((4, 4))
    Π[1, 2] = π_01_bb * πa_bb     # Prob(ϵ' = 1, z' = 0| ϵ = 0, z = 0) = Prob(ϵ' = 1| ϵ = 0, z' = 0, z = 0) * Prob(z' = 0| z = 0)
    Π[1, 1] = πa_bb - Π[1, 2]      # Π_bb00 + Π_bb01 = Π_bb
    Π[3, 4] = π_01_gg * πa_gg     # Prob(ϵ' = 1, z' = 1| ϵ = 0, z = 1) = Prob(ϵ' = 1| ϵ = 0, z' = 1, z = 1) * Prob(z' = 1| z = 1)
    Π[3, 3] = πa_gg - Π[3, 4]      # Π_gg00 + Π_gg01 = Π_gg

    # Conditions for transtion Π_gb00 / Π_gb = 1.25 Π_bb00 / Π_bb
    Π[3, 1] = 1.25 * Π[1, 1] / Πa_ag[1, 1] * Πa_ag[2, 1]
    Π[3, 2] = Πa_ag[2, 1] - Π[3, 1] # Π_gb00 + Π_gb01 = Π_gb

    # Conditions for transition Π_bg00 / Π_bg = 0.75 Π_gg00 / Π_gg
    Π[1, 3] = 0.75 * Π[3, 3] / Πa_ag[2, 2] * Πa_ag[1, 2]
    Π[1, 4] = Πa_ag[1, 2] - Π[1, 3] # Π_bg00 + Π_bg01 = Π_bg

    # Imposing the law of motion for unemployment
    # u_s * Π_ss'00 / Π_ss' + (1 - u_s) * Π_ss'10 / Π_ss' = u_s'
    Π[2, 1] = ur_b * πa_bb / (1 - ur_b) * (1 - Π[1, 1] / πa_bb)
    Π[2, 2] = πa_bb - Π[2, 1] # Π_bb10 + Π_bb11 = Π_bb
    Π[2, 3] = Πa_ag[1, 2] / (1 - ur_b) * (ur_g - ur_b * Π[1, 3] / Πa_ag[1, 2])
    Π[2, 4] = Πa_ag[1, 2] - Π[2, 3] # Π_bg10 + Π_bg11 = Π_bg
    Π[4, 1] = Πa_ag[2, 1] / (1 - ur_g) * (ur_b - ur_g * Π[3, 1] / Πa_ag[2, 1])
    Π[4, 2] = Πa_ag[2, 1] - Π[4, 1] # Π_gb10 + Π_gb11 = Π_gb
    Π[4, 3] = ur_g * πa_gg / (1 - ur_g) * (1 - Π[3, 3] / πa_gg)
    Π[4, 4] = πa_gg - Π[4, 3] # Π_gg10 + Π_gg11 = Π_gg

    # Generate full transition matrix with other shocks
    Π_full = kron(Πδ_ag, Πβ_ag, Π)
    Π_ag = kron(Πδ_ag, Πβ_ag, Πa_ag)

    return ur_b, er_b, ur_g, er_g, Π_full, Π_ag
end

@doc raw"""
    unpack(variable::String, npar::NumericalParametersAll)

Function that unpacks the relevant variable from the ravel array

# Arguments
- `variable::String`: variable to be unpacked
- `npar::NumericalParametersAll`: numerical parameters

# Returns
- `index_list::Array`: list of the relevant index
"""
function unpack(pos_array::Array{Tuple{Int64,Int64,Int64},3}, variable::String)
    # Identifying the relevant index
    if variable == "a"
        index = 1
    elseif variable == "β"
        index = 2
    elseif variable == "δ"
        index = 3
    else
        @error "Variable not found"
    end

    # Unpacking the variable
    index_list = []
    for j in eachindex(pos_array)
        push!(index_list, pos_array[j][index])
    end
    return index_list
end

function unpack_simulation(
    pos_array::Array{Tuple{Int64,Int64,Int64},3},
    simult_ag::Array{Int64,1},
)

    # Creating empty arrays to return
    a_list = []
    β_list = []
    δ_list = []

    # Iterating over indicies to unpack the relevant state of the economy
    for j in eachindex(simult_ag)
        push!(a_list, pos_array[simult_ag[j]][1])
        push!(β_list, pos_array[simult_ag[j]][2])
        push!(δ_list, pos_array[simult_ag[j]][3])
    end
    return a_list, β_list, δ_list
end
