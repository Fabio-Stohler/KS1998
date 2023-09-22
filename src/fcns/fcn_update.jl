function update_EVk(
        rU::Array,
        B::Array,
        mpar::ModelParameters,
        npar::NumericalParameters
    )
    EVk = similar(rU)
    Vk = zeros(npar.ngridk, npar.ngridkm, npar.nstates_ag, npar.nstates_id, npar.nstates_ag)
    update_EVk!(
        EVk,
        Vk,
        rU,
        B,
        mpar,
        npar)
    return EVk
end

function update_EVk!(
    EVk::Array,
    Vk::Array,
    rmu::Array,
    B::Array,
    mpar::ModelParameters,
    npar::NumericalParameters
    )
    # Export necessary arrays
    km = npar.km
    Π = npar.Π

    # Interpolate on tomorrows capital stock
    km_prime = exp.(B[:,1] .+ B[:, 2] * log.(km)')
    # Conditional on todays productivity grid ZZ, interpolate onto tomorrows capital grid
    # Dependence follows from the LOM which depends on ZZ
    for yy in 1:npar.nstates_id # Tomorrows income grid
        for zz in 1:npar.nstates_ag # Tomorrows productivity
            for kk in 1:npar.ngridk # Tomorrows capital grid 
                for ZZ in 1:npar.nstates_ag # Todays productivity
                    # Interpolate on tomorrows capital stock 
                    Vk[kk, :, zz, yy, ZZ] = mylinearinterpolate(
                        km,
                        rmu[kk, :, zz, yy],
                        km_prime[ZZ,:]
                    )
                    # mylinearinterpolate!(
                        # Vk[kk, :, zz, yy, ZZ],
                        # km, 
                        # rmu[kk, :, zz, yy], 
                        # km_prime[ZZ,:]
                    # )
                    # Cubic Spline interpolation
                    #KS.evaluate(
                    #    KS.Spline1D(
                    #        km, 
                    #        rU[kk, :, zz, yy],
                    #        bc = "extrapolate"), 
                    #    exp.(B[ZZ,1] .+ B[ZZ, 2] .* log.(km))
                        #)
                end
            end
        end
    end
    

    # Taking expectations over the marginal value
    for kk in 1:npar.ngridk # Current individal capital 
        for km in 1:npar.ngridkm # Current aggregate capital
            for yy in 1:npar.nstates_id # Current income state
                for zz in 1:npar.nstates_ag # Current aggregate productivity
                    # For each (individual) state today there exist four states tomorrow
                    EVk[kk, km, zz, yy] = dot(Vk[kk, km, :, :, zz]'[:], Π[(zz.-1)*2+yy,:])
                end
            end
        end
    end
end