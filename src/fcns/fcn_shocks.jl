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
    else
        return shocks_parameters_depreciation()
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
    shocks_parameters_depreciation()

Function that generates the transition matrix for the aggregate shocks and the transition matrix conditional on the aggregate depreciation state

# Returns
- `ur_b::Float64`: unemployment rate in a bad aggregate state
- `er_b::Float64`: employment rate in a bad aggregate state
- `ur_g::Float64`: unemployment rate in a good aggregate state
- `er_g::Float64`: employment rate in a good aggregate state
- `Π::Array`: transition matrix conditional on the aggregate state
- `Π_ag::Array`: transition matrix for the aggregate state
"""

function shocks_parameters_depreciation()
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
