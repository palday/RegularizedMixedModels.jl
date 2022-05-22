using Documenter
using RegularizedMixedModels

makedocs(;
    sitename="RegularizedMixedModels",
    doctest=true,
    pages=[
        "index.md",
        "api.md",
    ],
)

deploydocs(;repo = "github.com/palday/RegularizedMixedModels.jl.git", push_preview = true, devbranch = "main")
