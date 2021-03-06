# these should be split into things that take type parameters vs. those that don't
_delegations = [:(StatsBase.coefnames),
                # :(StatsBase.deviance),
                :(StatsBase.loglikelihood),
                :(LinearAlgebra.cond),
                :(StatsBase.dof),
                :(StatsBase.dof_residual),
                :(MixedModels.issingular),
                :(StatsBase.isfitted),
                :(StatsBase.meanresponse),
                :(StatsBase.modelmatrix),
                :(StatsBase.nobs),
                :(StatsBase.predict),
                :(MixedModels.raneftables),
                :(StatsBase.residuals),
                :(Base.size),
                :(StatsBase.response),
                :(MixedModels.σs),
                :(MixedModels.σρs),
                :(StatsBase.vcov),
                :(StatsModels.formula),
                :(StatsBase.coef),
                :(StatsBase.coeftable),
                :(MixedModels.condVar),
                :(MixedModels.condVartables),
                :(MixedModels.βs),
                :(GLM.dispersion),
                :(GLM.dispersion_parameter),
                :(MixedModels.feL),
                :(StatsBase.fitted),
                :(MixedModels.fixef),
                :(MixedModels.fixefnames),
                :(MixedModels.fnames),
                :(MixedModels.getθ),
                :(MixedModels.lowerbd),
                :(MixedModels.nθ),
                :(StatsBase.islinear),
                :(StatsBase.leverage),
                :(MixedModels.pwrss),
                :(MixedModels.ranef),
                :(MixedModels.rePCA),
                :(MixedModels.PCA),
                :(MixedModels.reweight!),
                :(MixedModels.varest),
                :(MixedModels.sdest),
                :(MixedModels.setθ!),
                :(MixedModels.sparseL),
                :(MixedModels.ssqdenom),
                :(Statistics.std),
                :(StatsBase.stderror),
                :(MixedModels.updateL!),
                :(StatsBase.weights)]

for f in _delegations
    @eval $f(m::RegularizedLinearMixedModel, args...; kwargs...) = $f(m.lmm, args...; kwargs...)
end
