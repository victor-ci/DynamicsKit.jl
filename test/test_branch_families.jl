@testset "Branch family inference" begin
    using Dates: DateTime
    sys = henon_map()
    function synthetic_branch(points; period=1, param_name=:a)
        branch_points = [
            (param=p, x1=x, x2=y, stable=true)
            for (p, x, y) in points
        ]
        BranchResult(
            CombinedBranchResult(branch_points, Any[]),
            period,
            sys.name,
            param_name,
            DateTime(2026, 1, 1)
        )
    end

    branch_a = synthetic_branch([(0.1, 1.0, 0.0), (0.2, 1.1, 0.0), (0.3, 1.2, 0.0)])
    branch_b = synthetic_branch([(0.1, 1.01, 0.0), (0.2, 1.11, 0.0), (0.3, 1.21, 0.0)])
    branch_c = synthetic_branch([(0.1, -1.0, 0.0), (0.2, -1.1, 0.0), (0.3, -1.2, 0.0)])

    @test DynamicsKit._branch_family_sample_indices(10, 7) == [1, 2, 4, 5, 7, 8, 10]
    @test DynamicsKit._branch_family_sample_indices(3, 9) == [1, 2, 3]

    assignments = branch_family_assignments(
        sys,
        [branch_a, branch_b, branch_c];
        params=[1.4, 0.3],
        sample_count=3,
        distance_tolerance=0.03
    )

    @test length(assignments) == 3
    @test assignments[1].family_id == assignments[2].family_id
    @test assignments[1].family_id != assignments[3].family_id
    @test assignments[1].family_label == "Family 1"
    @test assignments[3].diagnostics["sampleCount"] == 3

    different_period = branch_family_assignments(
        sys,
        [branch_a, synthetic_branch([(0.1, 1.0, 0.0), (0.2, 1.1, 0.0), (0.3, 1.2, 0.0)]; period=2)];
        params=[1.4, 0.3],
        sample_count=3,
        distance_tolerance=10.0
    )
    @test different_period[1].family_id != different_period[2].family_id

    nonoverlapping = branch_family_assignments(
        sys,
        [branch_a, synthetic_branch([(0.5, 1.0, 0.0), (0.6, 1.1, 0.0), (0.7, 1.2, 0.0)])];
        params=[1.4, 0.3],
        sample_count=3,
        distance_tolerance=10.0
    )
    @test nonoverlapping[1].family_id != nonoverlapping[2].family_id

    empty_branch = BranchResult(CombinedBranchResult(Any[], Any[]), 1, sys.name, :a, DateTime(2026, 1, 1))
    empty_assignments = branch_family_assignments(sys, [empty_branch]; params=[1.4, 0.3])
    @test length(empty_assignments) == 1
    @test empty_assignments[1].family_id == "family-1"
    @test empty_assignments[1].diagnostics["sampleCount"] == 0

    @test_throws ErrorException branch_family_assignments(
        sys,
        [synthetic_branch([(0.1, 1.0, 0.0)]; param_name=:missing)];
        params=[1.4, 0.3]
    )
end
