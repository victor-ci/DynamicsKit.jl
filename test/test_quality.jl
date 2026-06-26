@testset "Package quality" begin
    # `persistent_tasks` spawns a subprocess that re-precompiles the package to check for lingering
    # tasks after load — minutes of redundant precompile (and a CI timeout under load) for a compute
    # library that starts no background tasks. The remaining Aqua checks are the valuable ones here.
    Aqua.test_all(DynamicsKit; ambiguities=false, persistent_tasks=false)
end

