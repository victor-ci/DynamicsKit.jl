"""
Result I/O: save and load bifurcation results.
"""

"""
    save_result(filepath::String, result)

Save a bifurcation result to a JLD2 file.
"""
function save_result(filepath::String, result)
    dir = dirname(filepath)
    !isempty(dir) && !isdir(dir) && mkpath(dir)
    jldsave(filepath; result=result)
end

"""
    load_result(filepath::String)

Load a bifurcation result from a JLD2 file.
"""
function load_result(filepath::String)
    JLD2.load(filepath, "result")
end

