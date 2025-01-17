@testset "interest" begin
    # Extract the parameters
    mpar = KS.ModelParameters()
    npar = KS.NumericalParameters()
    nparδ = KS.NumericalParametersDelta()

    # Test multiple dispatch
    K = [1.0]
    Z = [1.0]
    L = [1.0]
    δ = [0.02]

    # Test the values the function returns
    @test KS.interest(K, Z, L, mpar) ≈ [mpar.α .- mpar.δ]
    @test KS.interest(K, Z[1], L, δ, mpar) ≈ mpar.α .- δ
end
