# Extension-source coverage that runs in every CPU-only test environment (no CUDA/AMDGPU/Metal
# hardware, and no need to install the huge CUDA/AMDGPU stacks as hard test dependencies). This does
# *not* prove the extensions actually work against real hardware — that requires the vendor package and
# a device, which `test_gpu_continuous.jl`'s device-gated block exercises when available. What this file
# *does* prove, unconditionally:
#   - every `ext/DynamicsKit*Ext.jl` file is syntactically valid Julia (parses cleanly);
#   - each defines the module name declared for it in `Project.toml`'s `[extensions]` table;
#   - each registers methods for the exact three extension points (`_dynamicskit_gpu_available`,
#     `_dynamicskit_gpu_unavailable_reason`, `_dynamicskit_gpu_backend`) on its own vendor `Val`, so a
#     future edit that silently drops one of these (a real defect: `gpu_backend(:vendor)` would then
#     throw the generic "no extension loaded" `ArgumentError` even with the vendor package installed)
#     fails this test instead of only failing on someone's GPU machine.
# Metal is additionally a hard test dependency (`[targets] test = [..., "Metal", ...]`), so its
# extension *does* load for real in every CI run; `test_gpu_backend.jl` already covers that. CUDA/AMDGPU
# are weak deps only — this file is the only coverage of their extension source in a CPU-only CI run.

using TOML

@testset "GPU extension source coverage (CPU-only, no vendor hardware required)" begin
    # (owner module name, extension file, vendor Symbol)
    extensions = [
        ("DynamicsKitCUDAExt", "DynamicsKitCUDAExt.jl", :cuda),
        ("DynamicsKitAMDGPUExt", "DynamicsKitAMDGPUExt.jl", :amdgpu),
        ("DynamicsKitMetalExt", "DynamicsKitMetalExt.jl", :metal),
    ]
    ext_dir = joinpath(pkgdir(DynamicsKit), "ext")

    # Unwrap `@doc "..." module Foo ... end` (a docstring macrocall wrapping the module expression) down
    # to the bare `module` expression, so the walk below doesn't need to special-case the docstring.
    function _unwrap_module_expr(ast::Expr)
        if ast.head == :module
            return ast
        end
        if ast.head == :macrocall
            for arg in ast.args
                arg isa Expr && (found = _unwrap_module_expr(arg); found !== nothing && return found)
            end
        end
        return nothing
    end
    _unwrap_module_expr(::Any) = nothing

    # Fully-qualified call names appear as `Expr(:., :DynamicsKit, QuoteNode(:_dynamicskit_gpu_available))`;
    # bare names appear as plain `Symbol`s. Normalize both to a `Symbol`.
    function _call_name(expr)
        expr isa Symbol && return expr
        if expr isa Expr && expr.head == :.
            last_arg = expr.args[end]
            return last_arg isa QuoteNode ? last_arg.value : nothing
        end
        return nothing
    end

    # From a `::Val{:vendor}` argument expression, extract `:vendor`; `nothing` if it doesn't match.
    function _val_symbol(arg)
        arg isa Expr && arg.head == :(::) || return nothing
        typeexpr = arg.args[end]
        typeexpr isa Expr && typeexpr.head == :curly && length(typeexpr.args) == 2 || return nothing
        typeexpr.args[1] == :Val || return nothing
        vendor = typeexpr.args[2]
        return vendor isa QuoteNode ? vendor.value : nothing
    end

    # Walk every top-level statement in the module body and collect (function_name, vendor_symbol) pairs
    # for both long-form (`function f(...) ... end`) and short-form (`f(...) = ...`) definitions.
    function _collect_registrations(module_body::Expr)
        found = Tuple{Symbol, Symbol}[]
        for stmt in module_body.args
            stmt isa Expr || continue
            call_expr = nothing
            if stmt.head == :function && !isempty(stmt.args)
                call_expr = stmt.args[1]
            elseif stmt.head == :(=) && !isempty(stmt.args) && stmt.args[1] isa Expr && stmt.args[1].head == :call
                call_expr = stmt.args[1]
            end
            call_expr === nothing && continue
            call_expr isa Expr && call_expr.head == :call || continue
            fname = _call_name(call_expr.args[1])
            fname === nothing && continue
            for arg in call_expr.args[2:end]
                vendor = _val_symbol(arg)
                vendor === nothing && continue
                push!(found, (fname, vendor))
            end
        end
        return found
    end

    required_methods = (:_dynamicskit_gpu_available, :_dynamicskit_gpu_unavailable_reason, :_dynamicskit_gpu_backend)

    for (module_name, filename, vendor) in extensions
        path = joinpath(ext_dir, filename)
        @testset "$(filename)" begin
            @test isfile(path)
            source = read(path, String)

            # Syntax check: parses cleanly as Julia, independent of the vendor package being installed.
            ast = Meta.parseall(source; filename=filename)
            has_syntax_error = any(a -> a isa Expr && a.head == :error, ast.args)
            @test !has_syntax_error

            module_expr = nothing
            for arg in ast.args
                arg isa Expr || continue
                module_expr = _unwrap_module_expr(arg)
                module_expr !== nothing && break
            end
            @test module_expr !== nothing

            if module_expr !== nothing
                # Module name must exactly match the `[extensions]` mapping in Project.toml, or Julia's
                # extension mechanism silently never loads it for the declared weak dependency.
                @test module_expr.args[2] == Symbol(module_name)

                module_body = module_expr.args[3]
                registrations = _collect_registrations(module_body)
                for method in required_methods
                    @test (method, vendor) in registrations
                end

                # No registration for a *different* vendor Val leaking into this file (a copy/paste
                # defect between the three near-identical extensions).
                other_vendor_hits = [reg for reg in registrations if reg[2] != vendor]
                @test isempty(other_vendor_hits)
            end
        end
    end

    @testset "Project.toml [extensions] table matches ext/ filenames and weakdeps" begin
        project_toml = joinpath(pkgdir(DynamicsKit), "Project.toml")
        toml = TOML.parsefile(project_toml)
        package_name_by_vendor = Dict(:cuda => "CUDA", :amdgpu => "AMDGPU", :metal => "Metal")

        @test haskey(toml, "extensions")
        @test haskey(toml, "weakdeps")
        extensions_table = toml["extensions"]
        weakdeps_table = toml["weakdeps"]

        for (module_name, filename, vendor) in extensions
            package_name = package_name_by_vendor[vendor]
            # Exact mapping, not merely present: `[extensions]` must map this module name to precisely
            # the vendor package Julia's extension mechanism loads it against.
            @test get(extensions_table, module_name, nothing) == package_name
            # The extension's weak dependency must be declared, or the mapping above is unloadable.
            @test haskey(weakdeps_table, package_name)
            @test isfile(joinpath(ext_dir, filename))
        end

        # No stray/undeclared extension mappings beyond the three this file checks.
        @test Set(keys(extensions_table)) == Set(first.(extensions))
        # The three vendors this file checks are exactly DynamicsKit's known/enumerable GPU vendors.
        @test Set(last.(extensions)) == Set(DynamicsKit._KNOWN_GPU_VENDORS)
    end
end
