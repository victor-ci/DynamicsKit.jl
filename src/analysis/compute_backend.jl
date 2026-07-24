# Optional compute-backend selection for the embarrassingly-parallel 2D sweeps (`bifurcation_map`,
# `lyapunov_field`, `basins_of_attraction`). CPU stays the zero-overhead default; GPU acceleration is
# opt-in via a typed backend selector rather than a config boolean, so the accepted values are
# self-documenting and the surface can grow (new vendors) without a breaking config change.
#
# Vendor GPU support ships as package extensions (weak dependencies): loading `DynamicsKit` alone
# loads the vendor-neutral KernelAbstractions/DiffEqGPU infrastructure but no device runtime or
# vendor binaries, and a CPU-only install is fully functional. A vendor extension registers
# itself by adding methods to `_dynamicskit_gpu_available` / `_dynamicskit_gpu_backend` for its
# `Val{vendor}`; see `ext/DynamicsKitMetalExt.jl` for the (deliberately non-accelerating — see below)
# reference implementation.
#
# GPU kernels are written against `KernelAbstractions.jl`, which is a plain, lightweight, CPU-capable
# dependency (no GPU stack of its own) — this is what keeps the CPU-only install lightweight while the
# kernel bodies stay portable to any KernelAbstractions-backed vendor package.
#
# Apple/Metal note: Metal GPUs (and Metal.jl) have no double-precision (Float64) hardware support at
# all — MSL has no `double` type. This library's period/Lyapunov closure tolerances are specified and
# validated in Float64; silently downgrading the sweep to Float32 would change classification results
# near regime boundaries, which is exactly the "scientifically weak" GPU toggle this feature must not
# ship. `DynamicsKitMetalExt` therefore loads (so `using Metal` doesn't error) but always reports
# unavailable with that explicit reason — an explicit `gpu_backend(:metal)` request gets a clear,
# specific error rather than a silently wrong answer. CUDA and AMDGPU (both FP64-capable) are
# recognized vendor names ready for an extension once such hardware is available to validate against.

"""
    ComputeBackend

Abstract supertype for the backend selector accepted by GPU-eligible sweeps (`bifurcation_map`,
`lyapunov_field`, `basins_of_attraction`). Concrete backends: [`CPUBackend`](@ref) (default),
[`AutoBackend`](@ref), and [`GPUBackend`](@ref). Construct them with [`cpu_backend`](@ref),
[`auto_backend`](@ref), and [`gpu_backend`](@ref).
"""
abstract type ComputeBackend end

"""
    CPUBackend <: ComputeBackend

Run on the existing threaded pure-Julia sweep. The default for every GPU-eligible analysis; passing it
explicitly is equivalent to omitting `backend=` and carries no additional overhead.
"""
struct CPUBackend <: ComputeBackend end

"""
    AutoBackend <: ComputeBackend

Use a GPU backend if one is loaded, available, and the call is GPU-eligible; otherwise run on the CPU.
Never errors due to device unavailability — use [`GPUBackend`](@ref) to require a specific vendor.
"""
struct AutoBackend <: ComputeBackend end

"""
    GPUBackend{Vendor} <: ComputeBackend

Explicitly request GPU execution on vendor `Vendor` (a `Symbol`, e.g. `:cuda`, `:amdgpu`, `:metal`).
Construct with [`gpu_backend`](@ref). Unlike [`AutoBackend`](@ref), an explicit `GPUBackend` request
throws a clear `ArgumentError` if the vendor's extension is not loaded/available, or if the call is not
GPU-eligible — it never silently falls back to CPU.
"""
struct GPUBackend{Vendor} <: ComputeBackend end
GPUBackend(vendor::Symbol) = GPUBackend{vendor}()

"""    cpu_backend() -> CPUBackend()"""
cpu_backend() = CPUBackend()

"""    auto_backend() -> AutoBackend()"""
auto_backend() = AutoBackend()

"""
    gpu_backend(vendor::Symbol) -> GPUBackend{vendor}

Explicitly request GPU vendor `vendor` (e.g. `:cuda`, `:amdgpu`, `:metal`). Use
[`gpu_backend_available`](@ref) to check first, or pass the result straight to a `backend=` keyword and
let it raise its own explicit error if unavailable.
"""
gpu_backend(vendor::Symbol) = GPUBackend(vendor)

"""    gpu_vendor(backend::GPUBackend{Vendor}) -> Vendor"""
gpu_vendor(::GPUBackend{Vendor}) where Vendor = Vendor

# Extension points. A vendor extension (loaded via its weak dependency) adds methods to these for its
# own `Val{vendor}`; the defaults here describe "no extension loaded for this vendor" so every
# unimplemented/unavailable vendor behaves identically and explicitly rather than silently.
_dynamicskit_gpu_available(::Val) = false
_dynamicskit_gpu_unavailable_reason(::Val) = "no matching DynamicsKit GPU extension is loaded"
function _dynamicskit_gpu_backend(::Val{Vendor}) where Vendor
    throw(ArgumentError("No DynamicsKit GPU extension is loaded or available for vendor :$(Vendor)."))
end

# Optional vendor hook for runtime preparation needed only by continuous DiffEqGPU kernels. Discrete
# KernelAbstractions sweeps never call it. Vendor extensions may adjust process-local device runtime
# limits before the first adaptive ensemble launch.
_prepare_continuous_gpu_backend(ka_backend, trajectories::Int) = nothing

# Vendors DynamicsKit knows how to enumerate via `available_gpu_backends()`. This is not an allowlist —
# `gpu_backend(:anything)` is always constructible — it is only the set this package actively probes
# when listing/auto-selecting. `:cuda` and `:amdgpu` are FP64-capable and recognized as extension
# targets with package extensions. `:metal` ships an extension that always reports unavailable (see
# file header).
const _KNOWN_GPU_VENDORS = (:cuda, :amdgpu, :metal)

"""
    gpu_backend_available(vendor::Symbol) -> Bool
    gpu_backend_available(backend::GPUBackend) -> Bool

Whether GPU vendor `vendor` is currently usable: its extension is loaded (the corresponding package,
e.g. `Metal`, has been `using`-imported) *and* it reports a working, scientifically-suitable device.
"""
gpu_backend_available(vendor::Symbol) = _dynamicskit_gpu_available(Val(vendor))
gpu_backend_available(::GPUBackend{Vendor}) where Vendor = gpu_backend_available(Vendor)

"""
    available_gpu_backends() -> Vector{Symbol}

The recognized GPU vendors (see [`_KNOWN_GPU_VENDORS`]) that report as available right now. Empty on a
CPU-only install, or when no loaded vendor extension reports a usable device.
"""
available_gpu_backends() = Symbol[vendor for vendor in _KNOWN_GPU_VENDORS if gpu_backend_available(vendor)]

function _first_available_gpu_vendor()
    for vendor in _KNOWN_GPU_VENDORS
        gpu_backend_available(vendor) && return vendor
    end
    return nothing
end

# Validate serialized provenance before `Symbol` conversion because interned symbols are never collected.
const _SERIALIZABLE_COMPUTE_BACKEND_NAMES = Set{String}(String.((:cpu, _KNOWN_GPU_VENDORS..., :_ka_cpu_test)))

"""
    _compute_backend_symbol(name::AbstractString) -> Symbol

Convert an allowlisted serialized `computeBackend` name to a `Symbol`.
"""
function _compute_backend_symbol(name::AbstractString)
    name in _SERIALIZABLE_COMPUTE_BACKEND_NAMES || throw(ArgumentError(
        "Unknown computeBackend value $(repr(name)); expected one of " *
        "$(sort(collect(_SERIALIZABLE_COMPUTE_BACKEND_NAMES)))."
    ))
    return Symbol(name)
end

# Test-only seam: exercises the exact KernelAbstractions kernel-launch path (upload / kernel / copy-back
# / cache-hook merge) without requiring real GPU hardware, by resolving to KernelAbstractions' own CPU
# backend. Deliberately excluded from `_KNOWN_GPU_VENDORS` (and so from `available_gpu_backends()` /
# `auto_backend()`); a user would have to explicitly write `gpu_backend(:_ka_cpu_test)` to reach it,
# which the leading underscore marks as a private implementation detail, not a supported vendor.
_dynamicskit_gpu_available(::Val{:_ka_cpu_test}) = true
_dynamicskit_gpu_backend(::Val{:_ka_cpu_test}) = KernelAbstractions.CPU()

"""
Resolve a user-facing `ComputeBackend` to `(ka_backend, vendor)`: `ka_backend` is `nothing` to run the
existing CPU code path, or a concrete `KernelAbstractions.Backend` to launch the GPU kernel on; `vendor`
is the `Symbol` actually selected (`:cpu` when `ka_backend === nothing`) — this is the provenance value
sweeps record on their result.

`eligible` is the call-site GPU-eligibility check (e.g. fixed-seed traversal, no switching events); it
is evaluated regardless of backend so an explicit `GPUBackend` request on an ineligible call gets a
clear configuration error rather than silently running on the CPU. `analysis_name` / `requirement_text`
feed that error message.
"""
_resolve_gpu_backend(::CPUBackend, eligible::Bool, analysis_name::AbstractString, requirement_text::AbstractString) = (nothing, :cpu)

function _resolve_gpu_backend(::AutoBackend, eligible::Bool, analysis_name::AbstractString, requirement_text::AbstractString)
    eligible || return (nothing, :cpu)
    vendor = _first_available_gpu_vendor()
    vendor === nothing && return (nothing, :cpu)
    return (_dynamicskit_gpu_backend(Val(vendor)), vendor)
end

function _resolve_gpu_backend(backend::GPUBackend{Vendor}, eligible::Bool, analysis_name::AbstractString, requirement_text::AbstractString) where Vendor
    eligible || throw(ArgumentError(
        "GPU backend :$(Vendor) was explicitly requested for $(analysis_name), but this call is not " *
        "GPU-eligible: $(analysis_name) GPU acceleration requires $(requirement_text). " *
        "Use backend=CPUBackend() (the default) for this call."
    ))
    gpu_backend_available(Vendor) || throw(ArgumentError(
        "GPU backend :$(Vendor) was explicitly requested for $(analysis_name), but is not available " *
        "($(_dynamicskit_gpu_unavailable_reason(Val(Vendor)))). " *
        "Available GPU vendors in this session: $(available_gpu_backends())."
    ))
    return (_dynamicskit_gpu_backend(Val(Vendor)), Vendor)
end

"""
The fixed section-crossing root-find tolerance DiffEqGPU applies when it converts a
`ContinuousCallback` to a `GPUContinuousCallback` (`Base.convert` hard-codes `100*eps(Float32)`).
This is the absolute floor to which the continuous GPU path can localize a Poincaré crossing, so a
closure `precision` tighter than this cannot be guaranteed to match the CPU path — such calls are
rejected (explicit `GPUBackend`) or fall back to the CPU (`AutoBackend`) rather than silently running
with a weaker localization. Recorded on results as provenance.
"""
const _CONTINUOUS_GPU_ROOTFIND_ABSTOL = 100 * eps(Float32)

"""Whether a closure `precision` is coarser than (or equal to) the GPU section-localization floor."""
_continuous_gpu_precision_ok(precision::Real) = precision >= _CONTINUOUS_GPU_ROOTFIND_ABSTOL

"""
Resolve a `ComputeBackend` for a `ContinuousODE` sweep. Continuous GPU eligibility additionally
requires: the system carries a GPU out-of-place RHS (`has_continuous_gpu_rhs`), the closure precision
is no tighter than the section-localization floor (`_continuous_gpu_precision_ok`), and the
analysis-specific `structural_eligible` (fixed-seed traversal, no switching/multistability/linked
params, Lyapunov disabled, …). `CPUBackend`/`AutoBackend` never error on ineligibility; an explicit
`GPUBackend` raises a clear `ArgumentError` naming the unmet requirement.
"""
function _resolve_continuous_gpu_backend(backend::ComputeBackend, sys::ContinuousODE,
                                         structural_eligible::Bool, precision::Real,
                                         analysis_name::AbstractString,
                                         structural_requirement_text::AbstractString)
    has_rhs = has_continuous_gpu_rhs(sys)
    precision_ok = _continuous_gpu_precision_ok(precision)
    eligible = structural_eligible && has_rhs && precision_ok
    requirement = string(
        structural_requirement_text,
        "; a GPU out-of-place StaticArray right-hand side on the system (built-in continuous systems ",
        "provide one — imported/user systems that supply only an in-place RHS do not)",
        "; and a closure precision ≥ $(_CONTINUOUS_GPU_ROOTFIND_ABSTOL) (the fixed GPU section-crossing ",
        "localization floor, below which closure-based period detection cannot be guaranteed to match the CPU path)",
    )
    return _resolve_gpu_backend(backend, eligible, analysis_name, requirement)
end

"""
Reject a GPU request for the continuous Lyapunov field/diagram with its specific reason: it is a
*coupled two-trajectory* computation (two neighbouring trajectories integrated together, renormalized
and reprojected onto the Poincaré section at each return), not the embarrassingly-parallel
independent-trajectory pattern the continuous GPU ensemble path implements. `CPUBackend`/`AutoBackend`
run on the CPU; an explicit `GPUBackend` gets a clear `ArgumentError` naming this — not a blanket
"no ContinuousODE GPU".
"""
function _reject_continuous_lyapunov_gpu_backend(backend::ComputeBackend, analysis_name::AbstractString)
    _resolve_gpu_backend(backend, false, analysis_name,
        "an independent per-cell trajectory workload; the continuous Lyapunov field is instead a " *
        "coupled two-trajectory computation with per-return renormalization and section reprojection, " *
        "which the continuous GPU ensemble path does not implement (run it on the CPU, backend=CPUBackend())")
    return nothing
end

# Small host<->device transfer helpers shared by the GPU-kernel launch paths.
function _gpu_allocate_like(ka_backend, arr::AbstractArray{T}) where T
    return KernelAbstractions.allocate(ka_backend, T, size(arr)...)
end

function _gpu_upload(ka_backend, arr::AbstractArray{T}) where T
    dev = _gpu_allocate_like(ka_backend, arr)
    copyto!(dev, arr)
    return dev
end
