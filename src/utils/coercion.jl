"""
Value coercion + JSON-plain helpers shared across the library (analysis caches, result
serialization) and the workbench. Relocated here from `ui/workbench.jl` (Contract D / repository
split) because `analysis/` already depends on them — keeping the library self-contained.
"""

_plain(value::Nothing) = nothing
_plain(value::AbstractDict) = Dict{String, Any}(String(k) => _plain(v) for (k, v) in pairs(value))
_plain(value::AbstractMatrix) = [_plain(collect(row)) for row in eachrow(value)]
_plain(value::AbstractVector) = [_plain(v) for v in value]
_plain(value) = value

_as_string(value, default::String="") = isnothing(value) ? default : String(value)
_as_float(value, default::Float64=0.0) = isnothing(value) ? default : value isa Number ? Float64(value) : parse(Float64, strip(String(value)))
_as_int(value, default::Int=0) = isnothing(value) ? default : value isa Integer ? Int(value) : value isa Real ? Int(round(value)) : parse(Int, strip(String(value)))
_as_bool(value, default::Bool=false) = isnothing(value) ? default : value isa Bool ? value : lowercase(strip(String(value))) in ("1", "true", "yes", "on")

function _as_float_vector(value, default::Vector{Float64}=Float64[])
    isnothing(value) && return copy(default)
    if value isa AbstractString
        stripped = strip(value)
        isempty(stripped) && return Float64[]
        return [parse(Float64, strip(item)) for item in split(stripped, ',') if !isempty(strip(item))]
    elseif value isa AbstractVector
        return Float64[_as_float(item) for item in value]
    end
    return copy(default)
end

function _as_int_vector(value, default::Vector{Int}=Int[])
    isnothing(value) && return copy(default)
    if value isa AbstractString
        stripped = strip(value)
        isempty(stripped) && return Int[]
        return [parse(Int, strip(item)) for item in split(stripped, ',') if !isempty(strip(item))]
    elseif value isa AbstractVector
        return Int[_as_int(item) for item in value]
    end
    return copy(default)
end

function _jsonish_dict(value)
    value isa AbstractDict || return Dict{String, Any}()
    Dict{String, Any}(String(k) => _plain(v) for (k, v) in pairs(value))
end
