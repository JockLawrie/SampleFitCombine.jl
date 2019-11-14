using DataFrames
using Distributions
using Logging
using MLJ
using MLJBase
using StatsBase
using Optim

mutable struct NormalKDE <: MLJBase.Probabilistic
end

MLJBase.target_scitype(::Type{NormalKDE}) = AbstractVector{<:MLJ.Continuous}

function MLJBase.fit(model::NormalKDE, verbosity::Integer, X, y)
    n = length(y)
    W = 0.0  # running total weight
    A = 0.0  # running mean
    Q = 0.0  # running sum of squares (the numerator of the variance)
    for i = 1:n
        w  = X[i, :weight]
        w == 0.0 && continue
        yi = y[i]
        W += w
        Aprev = A
        A += w * (yi - A) / W
        Q += w * (yi - Aprev) * (yi - A)
    end
    [A, sqrt(Q/W)], nothing, nothing  # fitresult, cache, report
end

MLJBase.predict(model::NormalKDE, fitresult, Xnew) = Normal(fitresult[1], fitresult[2])

################################################################################

struct MyEnsemble{M<:Model}
    model::M
    K::Int
    samplingfraction::Float64
    components::Vector{Machine{M}}  # Trained components
    weights::Vector{Float64}        # Mixing weights (trained or pre-specified)
end

MyEnsemble(model, K, samplingfraction) = MyEnsemble(model, K, samplingfraction, Machine{typeof(model)}[], Float64[])

function fitcomponent!(ensemble::MyEnsemble, X, y, rows)
    length(ensemble.components) >= ensemble.K && error("Ensemble already has K=$(ensemble.K) trained components")
    mchn = machine(ensemble.model, X, y)
    fit!(mchn, rows=rows)
    push!(ensemble.components, mchn)
end


function combine!(ensemble::MyEnsemble; weights::Vector{Float64}=Float64[], loss::Function=(yhat,y) -> "hi")
    loss_provided = try
        if loss(1.0, 1.0) == "hi"
            false
        end
    catch e
        true
    end
    loss_provided    && return combine_optimal_weights!(ensemble, loss)
    isempty(weights) && return combine_uniform_weights!(ensemble)
    combine_prespecified_weights!(ensemble, weights)
end

function combine_prespecified_weights!(ensemble::MyEnsemble, w::Vector{Float64})
    # Checks
    isempty(ensemble.components) && @warn "Ensemble has no components to combine." && return
    nw  = length(w)
    nk  = length(ensemble.components)
    nw != nk && error("$(nw) weights were supplied to combine!, expecting $(nk) weights, 1 for each component.")

    # Ensure that ensemble.weights has nk elements
    weights = ensemble.weights
    nw      = length(weights)
    if nw < nk
        for k = (nw + 1):nk
            push!(weights, 0.0)
        end
    elseif nw > nk
        for k = 1:(nw - nk)
            pop!(weights)
        end
    end

    # Replace existing weights with new weights
    for k = 1:nk
        @inbounds weights[k] = w[k]
    end
    nothing
end

function combine_uniform_weights!(ensemble)
    empty!(ensemble.weights)
    nk = length(ensemble.components)
    w  = 1.0 / nk
    for k = 1:nk
        push!(ensemble.weights, w)
    end
    nothing
end

function combine_optimal_weights!(ensemble, loss)
    Xtrain = ensemble.components[1].args[1]
    ytrain = ensemble.components[1].args[2]
    logits = fill(0.0, ensemble.K - 1)
    w      = fill(0.0, ensemble.K)
    n      = size(Xtrain, 2)
    T      = typeof(predict(ensemble.components[1], Xtrain[1, :x]))
    pred_components = Vector{T}(undef, length(ensemble.components))
    function objective(b)
        result = 0.0
        computeweights!(w, b)
        combine_prespecified_weights!(ensemble, w)
        for i = 1:n
            yhat    = predict!(ensemble, Xtrain[i, :x], pred_components)
            result += loss(yhat, ytrain[i])  # Uniform weights
        end
        result
    end
    theta0 = copy(logits)
    opts   = Optim.Options(time_limit=60, f_tol=1e-12)   # Debug with show_trace=true
    mdl    = optimize(objective, theta0, LBFGS(), opts)  # TODO: optimize(objective, theta0, LBFGS(), opts; autodiff=:forward)
    theta1 = mdl.minimizer
    computeweights!(w, theta1)
    combine_prespecified_weights!(ensemble, w)
    nothing
end

function computeweights!(weights, logits)
    wtotal     = 1.0
    weights[1] = 1.0  # exp(0.0)
    K = length(weights)
    for k = 2:K
        @inbounds w = exp(logits[k-1])
        @inbounds weights[k] = w
        wtotal += w
    end
    weights ./= wtotal
end

function predict(ensemble::MyEnsemble, Xtest::Vector{Vector{Float64}})
    n = size(Xtest, 1)
    T = typeof(predict(ensemble.components[1], Xtest[1]))
    pred_components = Vector{T}(undef, length(ensemble.components))
    pred1     = predict!(ensemble, Xtest[1], pred_components)
    result    = Vector{typeof(pred1)}(undef, n)
    result[1] = pred1
    for i = 2:n
        result[i] = predict!(ensemble, Xtest[i], pred_components)
    end
    result
end

function predict!(ensemble::MyEnsemble, Xrow::Vector{Float64}, pred_components)
    for (k, mchn) in enumerate(ensemble.components)
        pred_components[k] = predict(mchn, Xrow)
    end
    MixtureModel(pred_components, ensemble.weights)
end

predict(mchn::Machine, Xnew) = MLJBase.predict(mchn.model, mchn.fitresult, Xnew)


############################
# Data: Y ~ 0.5*N(10, 2^2) + 0.5*N(15, 2^2)
N = 1000
X = DataFrame(weight=fill(1.0, N), x=fill([1.0], N))  # row = (weight=1.0, x=Vector{Float64})
y = vcat(10.0 .+ 2.0*randn(Int(0.5*N)), 15.0 .+ 2.0*randn(Int(0.5*N)))

# Ensemble
ensemble = MyEnsemble(NormalKDE(), 2, 0.8)

# Sample and fit
n       = Int(round(ensemble.samplingfraction * N))  # Sample size
w       = ProbabilityWeights([1.0/N for i = 1:N])    # Sample weights
i_all   = collect(1:N)  # Population row indices
i_train = fill(0, n)    # Sample row indices
for k = 1:ensemble.K
    wsample!(i_all, w, i_train; replace=true, ordered=true);  # ordered means sorted
    fitcomponent!(ensemble, X, y, i_train)
end

# Combine
combine!(ensemble)  # Uniform weights
println(ensemble.weights)

wt   = rand(ensemble.K)
wt ./= sum(wt)
combine!(ensemble; weights=wt)  # Pre-specified weights
println(ensemble.weights)

combine!(ensemble; loss=(yhat, y) -> -logpdf(yhat, y))  # Optimal weights
println(ensemble.weights)

#  Predict
Xtest = X[1:3, :x]
for mchn in ensemble.components
    println(predict(mchn, Xtest))
end
println(predict(ensemble, Xtest))


# Sequential weighting
ensemble = MyEnsemble(NormalKDE(), 2, 0.8)
n        = Int(round(ensemble.samplingfraction * N))  # Sample size
i_all    = collect(1:N)  # Population row indices
i_train  = fill(0, n)    # Sample row indices
loss     = (yhat, y) -> -logpdf(yhat, y)
fitcomponent!(ensemble, X, y, i_all)
for k = 2:ensemble.K
    reweight!(ensemble, loss)  # Reweight the sample according to loss (modifies X[:, :weight])
    fitcomponent!(ensemble, X, y, i_all)
end
combine!(ensemble; loss=loss)  # Optimal weights
println(ensemble.weights)
