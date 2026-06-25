@testset "Package quality" begin
    Aqua.test_all(DynamicsKit; ambiguities=false)
end

