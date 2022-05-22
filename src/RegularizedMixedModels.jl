module RegularizedMixedModels

using Base: Ryu

using LinearAlgebra
using GLM
using MixedModels
using NLopt
using Random
using StatsAPI
using StatsBase
using Statistics
using ProgressMeter

using LinearAlgebra: BlasFloat, BlasReal, HermOrSym, PosDefException, copytri!
using NLopt: Opt
using StatsBase: fit, fit!

using MixedModels: nθ, nlevs


export RegularizedMixedModel,
       RegularizedLinearMixedModel,
       Ridge,
       Lasso,
       ElasticNet

# things from MixedModels.jl

export @formula,
    AbstractReMat,
    Bernoulli,
    Binomial,
    BlockDescription,
    BlockedSparse,
    DummyCoding,
    EffectsCoding,
    Grouping,
    Gamma,
    GeneralizedLinearMixedModel,
    HelmertCoding,
    HypothesisCoding,
    IdentityLink,
    InverseGaussian,
    InverseLink,
    LinearMixedModel,
    LogitLink,
    LogLink,
    MixedModel,
    MixedModelBootstrap,
    Normal,
    OptSummary,
    Poisson,
    ProbitLink,
    RaggedArray,
    RandomEffectsTerm,
    ReMat,
    SeqDiffCoding,
    SqrtLink,
    UniformBlockDiagonal,
    VarCorr,
    aic,
    aicc,
    bic,
    coef,
    coefnames,
    coefpvalues,
    coeftable,
    columntable,
    cond,
    condVar,
    condVartables,
    deviance,
    dispersion,
    dispersion_parameter,
    dof,
    dof_residual,
    fit,
    fit!,
    fitted,
    fitted!,
    fixef,
    fixefnames,
    formula,
    fulldummy,
    fnames,
    GHnorm,
    isfitted,
    islinear,
    issingular,
    leverage,
    levels,
    logdet,
    loglikelihood,
    lowerbd,
    meanresponse,
    modelmatrix,
    model_response,
    nobs,
    objective,
    parametricbootstrap,
    pirls!,
    predict,
    pwrss,
    ranef,
    raneftables,
    rank,
    refarray,
    refit!,
    refpool,
    refvalue,
    replicate,
    residuals,
    response,
    responsename,
    restoreoptsum!,
    saveoptsum,
    shortestcovint,
    sdest,
    setθ!,
    simulate,
    simulate!,
    sparse,
    sparseL,
    std,
    stderror,
    updateL!,
    varest,
    vcov,
    weights,
    zerocorr


"""
    RegularizedMixedModel
"""
abstract type RegularizedMixedModel{T} <: MixedModel{T} end # model with fixed and random effects

include("rlmm.jl")
include("delegations.jl")

end # module
