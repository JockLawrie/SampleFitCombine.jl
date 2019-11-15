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

function combine_optimal_weights!(ensemble, lossfunc)
    Xtrain = ensemble.components[1].args[1]  # DataFrame
    Xtrain = Xtrain[!, :x]  # Vector{Vector{Float64}}
    ytrain = ensemble.components[1].args[2]
    logits = fill(0.0, length(ensemble.components) - 1)
    w      = fill(0.0, length(ensemble.components))
    n      = size(Xtrain, 1)
    T      = typeof(predict(ensemble.components[1], Xtrain[1]))
    pred_components = Vector{T}(undef, length(ensemble.components))
    function objective(b)
        result = 0.0
        computeweights!(w, b)
        combine_prespecified_weights!(ensemble, w)
        for i = 1:n
            yhat    = predict!(ensemble, Xtrain[i], pred_components)
            result += lossfunc(yhat, ytrain[i])  # Unweighted observations
        end
        result
    end
    theta0 = logits
    opts   = Optim.Options(time_limit=60, f_tol=1e-12)   # Debug with show_trace=true
    mdl    = optimize(objective, theta0, LBFGS(), opts;) # TODO: optimize(objective, theta0, LBFGS(), opts; autodiff=:forward)
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

predict(mchn::Machine, Xnew::Vector{Float64})         =  MLJBase.predict(mchn.model, mchn.fitresult, Xnew)
predict(mchn::Machine, Xnew::Vector{Vector{Float64}}) = [MLJBase.predict(mchn.model, mchn.fitresult, Xnew) for x in Xnew]

"Returns: lossfunc(predict(ensemble, X), y)"
function loss(ensemble, X, y, lossfunc)
    result = 0.0
    n = length(y)
    size(X, 1) != n && error("X and y do not have the same number of observations.")
    T = typeof(predict(ensemble.components[1], X[1]))
    pred_components = Vector{T}(undef, length(ensemble.components))
    for i = 1:n
        yhat    = predict!(ensemble, X[i], pred_components)
        result += lossfunc(yhat, y[i])
    end
    result
end

"Reweight the sample according to loss (modifies X[:, :weight])"
function reweight!(ensemble, X, y, lossfunc)
    x      = X[!, :x]  # Vector{Vector{Float64}}
    w      = X[!, :weight]
    n      = length(y)
    total  = 0.0
    T      = typeof(predict(ensemble.components[1], x[1]))
    pred_components = Vector{T}(undef, length(ensemble.components))
    for i = 1:n
        yhat   = predict!(ensemble, x[i], pred_components)
        L      = lossfunc(yhat, y[i])
        w[i]   = L
        total += L
    end
    total /= n  # Ensure that the weights sum to n
    w    ./= total
end


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
lossfunc = (yhat, y) -> -logpdf(yhat, y)
combine!(ensemble)  # Uniform weights
println(ensemble.weights)
println(loss(ensemble, X[!, :x], y, lossfunc))

wt   = rand(ensemble.K)
wt ./= sum(wt)
combine!(ensemble; weights=wt)  # Pre-specified weights
println(ensemble.weights)
println(loss(ensemble, X[!, :x], y, lossfunc))

combine!(ensemble; loss=lossfunc)  # Optimal weights
println(ensemble.weights)
println(loss(ensemble, X[!, :x], y, lossfunc))

#  Predict
Xtest = X[1:3, :x]
ytest = y[1:3]
for mchn in ensemble.components
    println(predict(mchn, Xtest))
end
println(predict(ensemble, Xtest))
println(loss(ensemble, Xtest, ytest, lossfunc))

# Sequential weighting
ensemble = MyEnsemble(NormalKDE(), 2, 0.8)
i_all    = collect(1:N)  # Population row indices
fitcomponent!(ensemble, X, y, i_all)
combine!(ensemble)       # Give the component a weight of 1.0
for k = 2:ensemble.K
    reweight!(ensemble, X, y, lossfunc)  # Reweight the sample according to loss (modifies X[:, :weight])
    fitcomponent!(ensemble, X, y, i_all)
end
combine!(ensemble; loss=lossfunc)  # Optimal weights
println(ensemble.weights)
for mchn in ensemble.components
    println(predict(mchn, X[1:1, :x]))
end
println(loss(ensemble, X[!, :x], y, lossfunc))
