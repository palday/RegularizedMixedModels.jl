using DataFrames
using MixedModels
using RegularizedMixedModels
using StableRNGs
using Test

using MixedModels: dataset

slp = DataFrame(dataset(:sleepstudy); copycols=true)
slp[!, :noise] = rand(StableRNG(42), nrow(slp))

lmm = fit(MixedModel, @formula(reaction ~ 1 + noise + days + (1 + days|subj)), slp)
rlmm = fit(RegularizedMixedModel, @formula(reaction ~ 1 + noise + days + (1 + days|subj)), slp, ElasticNet)
