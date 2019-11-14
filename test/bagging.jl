#=
Bootstrp Aggregation (Bagging).

INPUT:
- ytrain, Xtrain. Training data.
- ytest, Xtest. Test data.
- model. An instance of MLJ.Supervised.
- sampling_fraction. Float64. sample size = n = N * sampling_fraction,  where N = length(ytrain).
=#

using DataFrames
using Distributions
using Logging
using MLJ
using MLJBase
import MLJModels
import Distributions
import GLM
using MLJModels.GLM_
using Tables
using StatsBase
using Optim


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
    n      = size(Xtrain, 1)
    function objective(b)
        result = 0.0
        computeweights!(w, b)
        combine_prespecified_weights!(ensemble, w)
        for i = 1:n
            yhat    = predict(ensemble, view(Xtrain, i:i, :))[1]
            result += loss(yhat, ytrain[i])
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

function predict(ensemble::MyEnsemble, Xtest)
    n = size(Xtest, 1)
    v = view(Xtest, 1:1, :)
    pred1  = MixtureModel([predict(mchn, v)[1] for mchn in ensemble.components], ensemble.weights)
    result = Vector{typeof(pred1)}(undef, n)
    result[1] = pred1
    for i = 2:n
        v = view(Xtest, i:i, :)
        result[i] = MixtureModel([predict(mchn, v)[1] for mchn in ensemble.components], ensemble.weights)
    end
    result
end

predict(mchn::Machine, Xnew) = MLJBase.predict(mchn.model, mchn.fitresult, Xnew)

# TODO: predictrow for non-Probabilistic models


############################
# Data: Y ~ N(2 + 3x, 2^2), where x is Uniform(5, 15)
N = 1000
X = DataFrame(x1=5.0 .+ 10.0 .* rand(N))
y = 2.0 .+ 3.0 .* X[!, :x1] .+ 2.0 .* randn(N)

# Ensemble
ensemble = MyEnsemble(LinearRegressor(), 5, 0.8)

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
Xtest = view(X, 1:3, :)
for mchn in ensemble.components
    println(predict(mchn, Xtest))
end
println(predict(ensemble, Xtest))
