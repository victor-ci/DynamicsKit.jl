"""
Minimal dependency-free JSON pretty-printer shared by the `docs/benchmarks/`
scripts that write a JSON summary file. Kept dependency-free rather than
adding a JSON package to `docs/benchmarks/Project.toml` for what is only ever
a flat, human-readable summary printout.
"""

_json_escape(s::AbstractString) = replace(s, "\\" => "\\\\", "\"" => "\\\"", "\n" => "\\n")

_to_json(io::IO, x::AbstractString, indent::Int) = print(io, "\"", _json_escape(x), "\"")
_to_json(io::IO, x::Symbol, indent::Int) = _to_json(io, String(x), indent)
_to_json(io::IO, x::Bool, indent::Int) = print(io, x)
_to_json(io::IO, x::Union{Integer, AbstractFloat}, indent::Int) = print(io, isfinite(x) ? x : "null")
_to_json(io::IO, ::Nothing, indent::Int) = print(io, "null")

function _to_json(io::IO, x::AbstractDict, indent::Int)
    isempty(x) && return print(io, "{}")
    pad = "  "^(indent + 1)
    println(io, "{")
    entries = collect(x)
    for (i, (key, value)) in enumerate(entries)
        print(io, pad, "\"", _json_escape(string(key)), "\": ")
        _to_json(io, value, indent + 1)
        println(io, i == length(entries) ? "" : ",")
    end
    print(io, "  "^indent, "}")
end

function _to_json(io::IO, x::AbstractVector, indent::Int)
    isempty(x) && return print(io, "[]")
    pad = "  "^(indent + 1)
    println(io, "[")
    for (i, value) in enumerate(x)
        print(io, pad)
        _to_json(io, value, indent + 1)
        println(io, i == length(x) ? "" : ",")
    end
    print(io, "  "^indent, "]")
end

json_pretty_print(io::IO, x) = (_to_json(io, x, 0); println(io))
