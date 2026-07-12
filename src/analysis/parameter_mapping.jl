"""
Parameter mapping: maps swept parameter values into the full parameter vectors the analysis
kernels consume. Stable public API; the underscore names (`_inject_param`, `_build_params`, …)
remain as aliases for internal call sites.
"""

"""
    inject_param(base_params, index, value, linked_param_indices=Int[]) -> Vector{Float64}

Return a fresh parameter vector copied from `base_params` with `value` written at `index` and at
each `linked_param_indices`. Accepts any `AbstractVector{<:Real}` (e.g. `Vector`, `SVector`, a view)
and always returns a mutable `Vector{Float64}`. Non-mutating; length preserved.
"""
function inject_param(base_params::AbstractVector{<:Real}, index::Int, value,
                      linked_param_indices::AbstractVector{<:Integer}=Int[])
    p = Vector{Float64}(base_params)
    p[index] = value
    for linked_index in linked_param_indices
        p[linked_index] = value
    end
    return p
end

"""
    build_sweep_params(config::BruteForceConfig, varied_value) -> Vector{Float64}

Full parameter vector for a 1-D brute-force sweep at `varied_value`. When `config.fixed_params` is
empty the result is sized to `max(param_index, linked_param_indices...)` (zero-filled); otherwise it
is a copy of `fixed_params`. `varied_value` is written at `param_index` and all linked indices.
"""
function build_sweep_params(config::BruteForceConfig, varied_value::Float64)
    required = max(config.param_index, maximum(config.linked_param_indices; init=0))
    p = isempty(config.fixed_params) ? zeros(Float64, required) : copy(config.fixed_params)
    required <= length(p) || throw(ArgumentError(
        "build_sweep_params: param_index/linked_param_indices reference parameter slot $required, " *
        "but fixed_params has length $(length(p)). Provide a longer fixed_params or check the indices."))
    p[config.param_index] = varied_value
    for idx in config.linked_param_indices
        p[idx] = varied_value
    end
    return p
end

"""
    build_basins_params(config::BasinsConfig) -> Vector{Float64}

Parameter vector for a basins computation. Empty `fixed_params` ⇒ `[config.bif_param]`; otherwise a
copy of `fixed_params` with `config.param_index` set to `config.bif_param`.
"""
function build_basins_params(config::BasinsConfig)
    if isempty(config.fixed_params)
        return [config.bif_param]
    else
        p = copy(config.fixed_params)
        p[config.param_index] = config.bif_param
        return p
    end
end

"""
    basins_ic_template(sys, config::BasinsConfig) -> SVector

Full-state initial-condition template for the basins grid, validated to the system's state
dimension (`sys.dim`, the full state — not the projected `state_dim`). Empty `ic_template` ⇒
all-zeros. Throws `ArgumentError` for out-of-range / equal grid indices, or a mismatched template.
"""
function basins_ic_template(sys::DynamicalSystem, config::BasinsConfig)
    d = sys.dim
    (1 <= config.x_index <= d && 1 <= config.y_index <= d) ||
        throw(ArgumentError("Basins grid indices ($(config.x_index), $(config.y_index)) must lie in 1:$(d)."))
    config.x_index != config.y_index ||
        throw(ArgumentError("Basins grid indices must differ; both x_index and y_index are $(config.x_index), which would collapse the grid onto a 1-D line."))
    if isempty(config.ic_template)
        return zeros(SVector{d, Float64})
    end
    length(config.ic_template) == d ||
        throw(ArgumentError("BasinsConfig.ic_template has length $(length(config.ic_template)) but the system state dim is $(d)."))
    return SVector{d, Float64}(config.ic_template)
end

# Internal: minimum parameter-vector length needed to hold all a/b/linked indices.
function _map_required_param_len(config::BifurcationMapConfig)
    required_len = max(config.a_index, config.b_index, length(config.base_params))
    for idx in config.a_linked_param_indices
        required_len = max(required_len, idx)
    end
    for idx in config.b_linked_param_indices
        required_len = max(required_len, idx)
    end
    return required_len
end

"""
    map_param_template(config::BifurcationMapConfig) -> Vector{Float64}

Zero-padded base parameter buffer sized to hold all `a`/`b`/linked indices; existing `base_params`
values are preserved, padded entries are `0.0`.
"""
function map_param_template(config::BifurcationMapConfig)
    required_len = _map_required_param_len(config)
    p = isempty(config.base_params) ? zeros(required_len) : copy(config.base_params)
    if length(p) < required_len
        old_len = length(p)
        resize!(p, required_len)
        fill!(@view(p[(old_len + 1):required_len]), 0.0)
    end
    return p
end

"""    map_a_write_indices(config) -> Vector{Int}  — indices the first (a) axis writes."""
map_a_write_indices(config::BifurcationMapConfig) = [config.a_index; config.a_linked_param_indices]
"""    map_b_write_indices(config) -> Vector{Int}  — indices the second (b) axis writes."""
map_b_write_indices(config::BifurcationMapConfig) = [config.b_index; config.b_linked_param_indices]

# Internal: write a/b values into the given indices in place.
function _inject_map_params!(p::Vector{Float64}, a_indices::Vector{Int}, b_indices::Vector{Int}, a_val::Float64, b_val::Float64)
    @inbounds for idx in a_indices
        p[idx] = a_val
    end
    @inbounds for idx in b_indices
        p[idx] = b_val
    end
    return p
end

"""
    map_params_from_template(template, a_indices, b_indices, a_val, b_val) -> Vector{Float64}

Allocating: copy `template`, then inject `a_val`/`b_val` at the respective indices.
"""
function map_params_from_template(template::Vector{Float64},
                                  a_indices::Vector{Int},
                                  b_indices::Vector{Int},
                                  a_val::Float64,
                                  b_val::Float64)
    return _inject_map_params!(copy(template), a_indices, b_indices, a_val, b_val)
end

"""
    map_params_from_buffer!(buffer, template, a_indices, b_indices, a_val, b_val) -> buffer

Allocation-free hot-loop variant: overwrite `buffer` from `template`, then inject. Returns `buffer`.
"""
function map_params_from_buffer!(buffer::Vector{Float64},
                                 template::Vector{Float64},
                                 a_indices::Vector{Int},
                                 b_indices::Vector{Int},
                                 a_val::Float64,
                                 b_val::Float64)
    copyto!(buffer, template)
    return _inject_map_params!(buffer, a_indices, b_indices, a_val, b_val)
end

"""
    build_map_params(config::BifurcationMapConfig, a_val, b_val) -> Vector{Float64}

Convenience: one-call allocating build of the parameter vector for a 2-D `(a, b)` map point.
"""
function build_map_params(config::BifurcationMapConfig, a_val::Float64, b_val::Float64)
    return map_params_from_template(
        map_param_template(config),
        map_a_write_indices(config),
        map_b_write_indices(config),
        a_val,
        b_val
    )
end

# --- backward-compatible aliases (remove once all internal call sites use the public names) ---
const _inject_param            = inject_param
const _build_params            = build_sweep_params
const _build_basins_params     = build_basins_params
const _basins_ic_template      = basins_ic_template
const _map_param_template      = map_param_template
const _map_a_write_indices     = map_a_write_indices
const _map_b_write_indices     = map_b_write_indices
const _map_params_from_template = map_params_from_template
const _map_params_from_buffer! = map_params_from_buffer!
const _build_map_params         = build_map_params
