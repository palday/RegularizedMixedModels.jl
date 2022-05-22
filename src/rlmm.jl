"""
    RegularizedLinearMixedModel

"""
struct RegularizedLinearMixedModel{T<:AbstractFloat} <: RegularizedMixedModel{T}
    lmm::LinearMixedModel{T}
    penalty::Vector
    k::Vector{T}
    optsum::OptSummary{T}
    # β::Vector{T}
    # β₀::Vector{T}
    # θ::Vector{T}
    # b::Vector{Matrix{T}}
    # u::Vector{Matrix{T}}
    # u₀::Vector{Matrix{T}}
end

const Ridge = Function[Base.Fix2(norm, 2)]
const Lasso = Function[Base.Fix2(norm, 2)]
const ElasticNet = Function[Base.Fix2(norm, 1), Base.Fix2(norm, 2)]

function RegularizedLinearMixedModel(f::FormulaTerm, tbl, penalty=Ridge, k=ones(Float64, length(penalty)); kwargs...)
    lmm = LinearMixedModel(f, tbl; kwargs...)
    optsum = OptSummary([k; lmm.optsum.initial],
                        [0. .* k; lmm.optsum.lowerbd],
                        :LN_BOBYQA)
    return RegularizedLinearMixedModel(lmm, penalty, k, optsum)
end


function StatsAPI.fit(::Type{RegularizedLinearMixedModel}, f::FormulaTerm, args...;
    progress=true, REML=false, log=true, kwargs...)
    return fit!(RegularizedLinearMixedModel(f, args...; kwargs...); progress, REML, log)
end

function StatsAPI.fit(::Type{RegularizedMixedModel}, f::FormulaTerm, args...;
    progress=true, REML=false, log=true, kwargs...)
    return fit!(RegularizedLinearMixedModel(f, args...; kwargs...); progress, REML, log)
end

# function StatsBase.deviance(m::RegularizedLinearMixedModel{T}) where {T}
#     wts = m.sqrtwts
#     denomdf = T(ssqdenom(m))
#     σ = something(m.optsum.sigma, pwrss(m.lmm) / denomdf)
#     val = denomdf * (log2π + 2 * log(σ)) + logdet(m) + pwrss(m) / σ^2
#     return isempty(wts) ? val : val - T(2.0) * sum(log, wts)
# end

"""
    objective(m::RegularizedLinearMixedModel)

Return negative twice the log-likelihood of model `m` plus penalty
"""
function MixedModels.objective(m::RegularizedLinearMixedModel{T}) where {T}
    # this is useful for fitting given a particular k, but it's not good at finding
    # which k is best
    β = fixef(m.lmm)
    # need to add beta to search space
    return objective(m.lmm) + 2 * sum(k * p(β) for (k, p) in zip(m.k, m.penalty))
end

# need to do this like GLMM (no fast=true for now)
# searchspace is k-beta-theta
# then use PIRLS to evalute BLUPs?
# computed pwrss as sum(abs2, response .- fitted) + sum(abs2, ranef(m; uscale=true))^2 + sum(k * p(β) for (k, p) in zip(m.k, m.penalty))

function StatsAPI.fit!(
    m::RegularizedLinearMixedModel{T};
    REML=false,
    progress::Bool=true,
    log::Bool=true,
    kwargs...
) where {T}
    optsum = m.optsum
    lmm = m.lmm
    length(m.k) == length(m.penalty) || error("penalty weights and penalty functions have differnet lengths")
    lenk = length(m.k)
    lenθ = nθ(lmm)
    # this doesn't matter for LMM, but it does for GLMM, so let's be consistent
    if optsum.feval > 0
        throw(ArgumentError("This model has already been fitted. Use refit!() instead."))
    end
    if all(==(first(m.y)), m.y)
        throw(
            ArgumentError("The response is constant and thus model fitting has failed")
        )
    end
    opt = Opt(optsum)
    m.lmm.optsum.REML = optsum.REML = REML
    prog = ProgressUnknown("Minimizing"; showspeed=true)
    fitlog = optsum.fitlog
    function obj(x, g)
        isempty(g) || throw(ArgumentError("g should be empty for this objective"))
        k = view(x, 1:lenk)
        θ = view(x, (lenk+1):lastindex(x))
        val = try
            updateL!(setθ!(lmm, θ))
            objective(m)
        catch ex
            # This can happen when the optimizer drifts into an area where
            # there isn't enough shrinkage. Why finitial? Generally, it will
            # be the (near) worst case scenario value, so the optimizer won't
            # view it as an optimum. Using Inf messes up the quadratic
            # approximation in BOBYQA.
            ex isa PosDefException || rethrow()
            iter == 0 && rethrow()
            m.optsum.finitial
        end
        log && push!(fitlog, (copy(x), val))
        progress && ProgressMeter.next!(prog; showvalues=[(:objective, val), (:k, k)])
        return val
    end
    NLopt.min_objective!(opt, obj)
    # try
    optsum.finitial = obj(optsum.initial, T[])
    # catch ex
    #     ex isa PosDefException || rethrow()
    #     # give it one more try with a massive change in scaling
    #     @info "Initial step failed, rescaling initial guess and trying again."
    #     @warn """Failure of the initial step is often indicative of a model specification
    #              that is not well supported by the data and/or a poorly scaled model.
    #           """
    #     optsum.initial[(lenk+1):end] ./=
    #         (isempty(lmm.sqrtwts) ? 1.0 : maximum(lmm.sqrtwts)^2) *
    #         maximum(response(lmm))
    #     optsum.finitial = obj(optsum.initial, T[])
    # end
    empty!(fitlog)
    fmin, xmin, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    ProgressMeter.finish!(prog)
    ## check if small non-negative parameter values can be set to zero
    # should we exclude k here?
    xmin_ = copy(xmin)
    lb = optsum.lowerbd
    for i in eachindex(xmin_)
        if iszero(lb[i]) && zero(T) < xmin_[i] < T(0.001)
            xmin_[i] = zero(T)
        end
    end
    loglength = length(fitlog)
    if xmin ≠ xmin_
        if (zeroobj = obj(xmin_, T[])) ≤ (fmin + 1.e-5)
            fmin = zeroobj
            copyto!(xmin, xmin_)
        elseif length(fitlog) > loglength
            # remove unused extra log entr]y
            pop!(fitlog)
        end
    end
    ## ensure that the parameter values saved in m are xmin
    updateL!(setθ!(m, view(xmin, (lenk+1):lastindex(xmin))))

    optsum.feval = opt.numevals
    optsum.final = xmin
    optsum.fmin = fmin
    optsum.returnvalue = ret
    copyto!(m.k, view(optsum.final, 1:lenk))
    MixedModels._check_nlopt_return(ret)
    return m
end

function Base.getproperty(m::RegularizedLinearMixedModel{T}, s::Symbol) where {T}
    if s in fieldnames(RegularizedLinearMixedModel)
        getfield(m, s)
    else
        getproperty(m.lmm, s)
    end
end

function Base.propertynames(m::RegularizedLinearMixedModel, private::Bool=false)
    return (
        fieldnames(RegularizedLinearMixedModel)...,
        propertynames(LinearMixedModel, private)...
    )
end

# LinearAlgebra.rank(m::LinearMixedModel) = m.feterm.rank
"""
    refit!(m::LinearMixedModel[, y::Vector]; REML=m.optsum.REML, kwargs...)

Refit the model `m` after installing response `y`.

If `y` is omitted the current response vector is used.
`kwargs` are the same as [`fit!`](@ref).
"""
# function refit!(m::LinearMixedModel, y=response(m); kwargs...)
#     length(y) == length(response(m)) || throw(DimensionMismatch(""))
#     copyto!(resp, y)
#     return refit!(unfit!(m); kwargs...)
# end


function Base.setproperty!(m::RegularizedLinearMixedModel, s::Symbol, y)
    s in fieldnames(RegularizedLinearMixedModel)
    return s == :θ ? setθ!(m, y) : setfield!(m, s, y)
end

function Base.show(io::IO, ::MIME"text/plain", rlmm::RegularizedLinearMixedModel)
    if rlmm.optsum.feval < 0
        @warn("Model has not been fit")
        return nothing
    end
    m = rlmm.lmm
    println(io, "DISPLAYING JUST LMM STUFF FOR NOW")
    n, p, q, k = size(m)
    REML = m.optsum.REML
    println(io, "Linear mixed model fit by ", REML ? "REML" : "maximum likelihood")
    println(io, "Subject to regularization constrains with weights ", round.(rlmm.k; sigdigits=2))
    println(io, " ", m.formula)
    oo = objective(m)
    if REML
        println(io, " REML criterion at convergence: ", oo)
    else
        nums = Ryu.writefixed.([-oo / 2, oo, aic(m), aicc(m), bic(m)], 4)
        fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
        for label in ["  logLik", "-2 logLik", "AIC", "AICc", "BIC"]
            print(io, rpad(lpad(label, (fieldwd + textwidth(label)) >> 1), fieldwd))
        end
        println(io)
        print.(Ref(io), lpad.(nums, fieldwd))
        println(io)
    end
    println(io)

    show(io, VarCorr(m))

    print(io, " Number of obs: $n; levels of grouping factors: ")
    join(io, nlevs.(m.reterms), ", ")
    println(io)
    println(io, "\n  Fixed-effects parameters:")
    return show(io, coeftable(m))
end


function unfit!(rmodel::RegularizedLinearMixedModel{T}) where {T}
    model = rmodel.lmm
    model.optsum.feval = -1
    model.optsum.initial_step = T[]
    reevaluateAend!(model)

    return model
end
