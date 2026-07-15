using Test
using DynamicsKit
using Aqua
using StaticArrays
using LinearAlgebra

include("support/fixtures.jl")
using .TestFixtures

const ALL_TEST_FILES = [
    "test_quality.jl",
    "test_systems.jl",
    "test_parameter_mapping.jl",
    "test_result_accessors_contract.jl",
    "test_kernels_contract.jl",
    "test_cache_hook_contract.jl",
    "test_public_api_promotions.jl",
    "test_brute_force.jl",
    "test_lyapunov.jl",
    "test_lyapunov_spectrum.jl",
    "test_codim2.jl",
    "test_spectrum.jl",
    "test_continuation.jl",
    "test_collocation.jl",
    "test_map_special_points.jl",
    "test_branch_families.jl",
    "test_skeleton.jl",
    "test_atlas.jl",
    "test_basins_map_refine.jl",
]

const TEST_TARGETS = Dict(
    "quality" => ["test_quality.jl"],
    "systems" => ["test_systems.jl"],
    "parameter-mapping" => ["test_parameter_mapping.jl"],
    "parameter_mapping" => ["test_parameter_mapping.jl"],
    "accessors-contract" => ["test_result_accessors_contract.jl"],
    "accessors_contract" => ["test_result_accessors_contract.jl"],
    "kernels-contract" => ["test_kernels_contract.jl"],
    "kernels_contract" => ["test_kernels_contract.jl"],
    "cache-hook" => ["test_cache_hook_contract.jl"],
    "cache_hook" => ["test_cache_hook_contract.jl"],
    "public-api" => ["test_public_api_promotions.jl"],
    "public_api" => ["test_public_api_promotions.jl"],
    "brute-force" => ["test_brute_force.jl"],
    "brute_force" => ["test_brute_force.jl"],
    "lyapunov" => ["test_lyapunov.jl"],
    "lyapunov-spectrum" => ["test_lyapunov_spectrum.jl"],
    "lyapunov_spectrum" => ["test_lyapunov_spectrum.jl"],
    "codim2" => ["test_codim2.jl"],
    "spectrum" => ["test_spectrum.jl"],
    "continuation" => ["test_continuation.jl"],
    "collocation" => ["test_collocation.jl"],
    "map-special-points" => ["test_map_special_points.jl"],
    "map_special_points" => ["test_map_special_points.jl"],
    "branch-families" => ["test_branch_families.jl"],
    "branch_families" => ["test_branch_families.jl"],
    "skeleton" => ["test_skeleton.jl"],
    "atlas" => ["test_atlas.jl"],
    "basins-map-refine" => ["test_basins_map_refine.jl"],
    "basins_map_refine" => ["test_basins_map_refine.jl"],
)

function _selected_test_files(args)
    isempty(args) && return copy(ALL_TEST_FILES)
    selected = String[]
    for arg in args
        if arg in ("all", "full")
            return copy(ALL_TEST_FILES)
        end
        key = lowercase(String(arg))
        if !haskey(TEST_TARGETS, key)
            valid = join(sort(collect(keys(TEST_TARGETS))), ", ")
            error("Unknown test target '$arg'. Valid targets: $valid")
        end
        append!(selected, TEST_TARGETS[key])
    end
    return unique!(selected)
end

@testset "DynamicsKit" begin
    for file in _selected_test_files(ARGS)
        include(file)
    end
end
