using JSON3

struct RouterRecord
    block::Int
    slot::Int
    shape::String
    orientation::String
    experts::Int
    kind::String
    structural_name::String
end

function load_json_report(path::AbstractString)
    return JSON3.read(read(path, String))
end
