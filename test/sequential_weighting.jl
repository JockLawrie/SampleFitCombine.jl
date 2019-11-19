using Clustering
using DataFrames
using Distributions
using Logging
using MLJ
using MLJBase
using StatsBase
using Optim
using UnicodePlots

import Distributions.ncomponents

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

ncomponents(ensemble::MyEnsemble) = length(ensemble.components)

function fitcomponent!(ensemble::MyEnsemble, X, y; rows=collect(1:length(y)))
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

function optimize_weights!(ensemble, X, y, lossfunc)
    logits = fill(0.0, length(ensemble.components) - 1)
    w      = fill(0.0, length(ensemble.components))
    pred_components = construct_prediction_components(ensemble, X[1])
    function f(b)
        computeweights!(w, b)
        combine_prespecified_weights!(ensemble, w)
        loss!(pred_components, ensemble, X, y, lossfunc)
    end
    theta0 = logits
    opts   = Optim.Options(time_limit=60, f_tol=1e-12)   # Debug with show_trace=true
    mdl    = optimize(f, theta0, LBFGS(), opts;) # TODO: optimize(objective, theta0, LBFGS(), opts; autodiff=:forward)
    theta1 = mdl.minimizer
    computeweights!(w, theta1)
    combine_prespecified_weights!(ensemble, w)
    nothing
end

function construct_prediction_components(ensemble, Xtest::Vector{Float64})
    T = typeof(predict(ensemble.components[1], Xtest))
    Vector{T}(undef, length(ensemble.components))
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
    pred_components = construct_prediction_components(ensemble, Xtest[1])
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
function loss(ensemble, X::Vector{Vector{Float64}}, y::Vector{Float64}, lossfunc)
    size(X, 1) != length(y) && error("X and y do not have the same number of observations.")
    pred_components = construct_prediction_components(ensemble, X[1])
    loss!(pred_components, ensemble, X, y, lossfunc)
end

function loss!(pred_components, ensemble, X, y, lossfunc)
    result = 0.0
    n = length(y)
    for i = 1:n
        yhat    = predict!(ensemble, X[i], pred_components)
        result += lossfunc(yhat, y[i])
    end
    result / n
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
# Data: Y ~ 0.5*N(10, 2^2) + 0.5*N(20, 2^2)
N = 1000
X = DataFrame(weight=fill(1.0, N), x=fill([1.0], N))  # row = (weight=1.0, x=Vector{Float64})
y = vcat(10.0 .+ 2.0*randn(Int(0.5*N)), 20.0 .+ 2.0*randn(Int(0.5*N)))
histogram(y, nbins=30)

# Bagging
ensemble = MyEnsemble(NormalKDE(), 2, 0.8)
n        = Int(round(ensemble.samplingfraction * N))  # Sample size
w        = ProbabilityWeights([1.0/N for i = 1:N])    # Sample weights
i_all    = collect(1:N)  # Population row indices
i_train  = fill(0, n)    # Sample row indices
for k = 1:ensemble.K
    wsample!(i_all, w, i_train; replace=true, ordered=true);  # ordered means sorted
    fitcomponent!(ensemble, X, y; rows=i_train)
end
combine!(ensemble)  # Uniform weights
println(ensemble.weights)
lossfunc = (yhat, y) -> -logpdf(yhat, y)
println(loss(ensemble, X[!, :x], y, lossfunc))

# Combine components with pre-specified weights
wt   = rand(ensemble.K)
wt ./= sum(wt)
combine!(ensemble; weights=wt)
println(ensemble.weights)
println(loss(ensemble, X[!, :x], y, lossfunc))

# Combine components with optimized weights
optimize_weights!(ensemble, X[!, :x], y, lossfunc)
println(ensemble.weights)
println(loss(ensemble, X[!, :x], y, lossfunc))

# Predict
Xtest = X[1:3, :x]
ytest = y[1:3]
for mchn in ensemble.components
    println(predict(mchn, Xtest))
end
println(predict(ensemble, Xtest))
println(loss(ensemble, Xtest, ytest, lossfunc))

# Sequential weighting. Observations (y[i], X[i]) are reweighted proportionally to weightfunc(yhat[i], y[i])
ensemble   = MyEnsemble(NormalKDE(), 2, 0.8)
lossfunc   = (yhat, y) -> -logpdf(yhat, y)
weightfunc = lossfunc
fitcomponent!(ensemble, X, y)
combine!(ensemble)  # Give the component a weight of 1.0
for k = 2:ensemble.K
    reweight!(ensemble, X, y, weightfunc)  # Reweight the sample according to loss (modifies X[:, :weight])
    fitcomponent!(ensemble, X, y)
    combine!(ensemble; loss=lossfunc)  # Optimize weights
end
for mchn in ensemble.components
    println(predict(mchn, X[1:1, :x]))
end
println(ensemble.weights)
println(loss(ensemble, X[!, :x], y, lossfunc))

# Cyclic sequential weighting. Components are sequentially replaced with better-fitting components.
ensemble   = MyEnsemble(NormalKDE(), 2, 0.8)
lossfunc   = (yhat, y) -> -logpdf(yhat, y)
fitcomponent!(ensemble, X, y)
push!(ensemble.components, ensemble.components[1])
combine!(ensemble)  # Give the components equal weight
pred  = predict(ensemble, X[!, :x])
probs = [cdf(pred[i], y[i]) for i = 1:N]
probs = reshape(probs, 1, N)
res = kmeans(probs, 2)
a = assignments(res)
for cluster = 1:2
    # Remove the first component and shift the remaining components leftward
    popfirst!(ensemble.components)
    popfirst!(ensemble.weights)
    ensemble.weights ./= sum(ensemble.weights)

    # Sample
    sample = a .== cluster
    ytrain = y[sample]
    Xtrain = X[sample, :]

    # Fit
    fitcomponent!(ensemble, Xtrain, ytrain)
end
optimize_weights!(ensemble, X[!, :x], y, lossfunc)  # Combine
println(ensemble.weights)
for mchn in ensemble.components
    println(predict(mchn, X[1:1, :x]))
end
println(loss(ensemble, X[!, :x], y, lossfunc))


# Sequential weighting with unknown number of components.
# Replace the worst-fitting component with 2 components
function sequentialweight!(ensemble, X, y, lossfunc)
    fitcomponent!(ensemble, X, y)
    combine!(ensemble)  # Give the component a weight of 1
    Lmin = loss(ensemble, X[!, :x], y, lossfunc)  # Minimum loss achieved so far
    N    = length(y)
    a    = fill(1, N)   # Cluster assignments
    for itr = 1:(ensemble.K - 1)
        # Identify the worst-fitting component
        k_remove = 0
        maxloss  = 0.0
        for k = 1:ncomponents(ensemble)
            sample = a .== k
            Xsub = X[sample, :]
            ysub = y[sample]
            L    = loss(ensemble, Xsub[!, :x], ysub, lossfunc)  # Minimum loss achieved so far
            if L > maxloss
                k_remove = k
                maxloss  = L
            end
        end

        # Identify 2 clusters
        sample = a .== k_remove
        Xsub   = X[sample, :]
        ysub   = y[sample]
        Nsub   = length(ysub)
        pred   = predict(ensemble.components[k_remove], Xsub[!, :x])
        probs  = reshape([cdf(pred[i], ysub[i]) for i = 1:Nsub], 1, Nsub)
        res    = kmeans(probs, 2)
        a2     = assignments(res)

        # Remove the worst-fitting component (and shift the remaining components leftward)
        old_component = splice!(ensemble.components, k_remove)
        old_weights   = copy(ensemble.weights)
        splice!(ensemble.weights, k_remove)
        ensemble.weights ./= sum(ensemble.weights)
        a[a .== k_remove] .= 0   # Update cluster assignments
        for k = (k_remove + 1):(ncomponents(ensemble) + 1)
            a[a .== k] .= k - 1  # Shift higher clusters down
        end
        k1 = ncomponents(ensemble) + 1
        k2 = ncomponents(ensemble) + 2
        v = view(a, a .== 0)
        v[a2 .== 1] .= k1
        v[a2 .== 2] .= k2

        # Fit 2 replacement components
        for cluster = 1:2
            # Sample
            sample = cluster == 1 ? a .== k1 : a .== k2
            ytrain = y[sample]
            Xtrain = X[sample, :]

            # Fit
            fitcomponent!(ensemble, Xtrain, ytrain)
        end
        optimize_weights!(ensemble, X[!, :x], y, lossfunc)  # Combine

        # Stopping criteria
        L = loss(ensemble, X[!, :x], y, lossfunc)
        @info "Loss = $(L). Lmin = $(Lmin)"
        if L > Lmin || abs((L - Lmin)/Lmin) <= 0.005
            splice!(ensemble.components, k2)
            splice!(ensemble.components, k1)
            push!(ensemble.components, old_component)
            combine!(ensemble; weights=old_weights)
            break
        end
        Lmin = L
        ncomponents(ensemble) == ensemble.K && break
    end
    Lmin
end

# Data: Y ~ 0.333*N(10, 2^2) + 0.333*N(20, 2^2) + 0.333*N(30, 2^2)
N = 1500
X = DataFrame(weight=fill(1.0, N), x=fill([1.0], N));  # row = (weight=1.0, x=Vector{Float64})
y = vcat(10.0 .+ 2.0*randn(round(Int(N/3))), 17.0 .+ 2.0*randn(round(Int(N/3))), 24.0 .+ 2.0*randn(round(Int(N/3))));
#y = vcat(10.0 .+ 2.0*randn(round(Int(N/3))), 20.0 .+ 2.0*randn(round(Int(N/3))), 30.0 .+ 2.0*randn(round(Int(N/3))));
histogram(y, nbins=50)

# Sample-Fit-Combine
ensemble = MyEnsemble(NormalKDE(), 10, 0.8)  # Maximum of 10 components
lossfunc = (yhat, y) -> -logpdf(yhat, y)
sequentialweight!(ensemble, X, y, lossfunc)

# Results
println(ensemble.weights)
for mchn in ensemble.components
    println(predict(mchn, X[1:1, :x]))
end
println(loss(ensemble, X[!, :x], y, lossfunc))


"""
1. Start with K = 1 cluster and all observations assigned to the cluster.
2. Measure loss.
3. If another cluster is warranted (stopping criteria not satisfied):
   - Set K = K + 1
   - Identify K clusters in probs, and assign each observation to a cluster.
   - Remove all existing components and weights
   - For k = 1:K
         Set Xtrain, ytrain the observations in cluster k
         Estimate a new component on Xtrain, ytrain
"""
function clustercomponents!(ensemble, X, y, lossfunc)
    # Fit first cluster
    fitcomponent!(ensemble, X, y)
    combine!(ensemble)  # Give the component a weight of 1

    # Fit clusters 2:K
    Lmin = loss(ensemble, X[!, :x], y, lossfunc)  # Minimum loss achieved so far
    N    = length(y)
    for k = 2:ensemble.K
        pred  = predict(ensemble, X[!, :x])
        probs = reshape([cdf(pred[i], y[i]) for i = 1:N], 1, N)
        res   = kmeans(probs, k)
        a     = assignments(res)
        old_components = copy(ensemble.components)
        old_weights    = copy(ensemble.weights)
        for k2 = 1:(k-1)
            popfirst!(ensemble.components)
        end
        for cluster = 1:k
            # Sample
            sample = a .== cluster
            ytrain = y[sample]
            Xtrain = X[sample, :]

            # Fit
            fitcomponent!(ensemble, Xtrain, ytrain)
        end
        optimize_weights!(ensemble, X[!, :x], y, lossfunc)  # Combine

        # Stopping criteria
        L = loss(ensemble, X[!, :x], y, lossfunc)
        @info "K = $(k). Loss = $(L). Lmin = $(Lmin)"
        if L > Lmin || abs((L - Lmin)/Lmin) <= 0.005
            for k2 = 1:k
                popfirst!(ensemble.components)
            end
            for k2 = 1:length(old_components)
                push!(ensemble.components, old_components[k2])
            end
            combine!(ensemble; weights=old_weights)
            break
        end
        Lmin = L
        ncomponents(ensemble) == ensemble.K && break
    end
    Lmin
end

# Data: Y ~ 0.333*N(10, 2^2) + 0.333*N(20, 2^2) + 0.333*N(30, 2^2)
N = 1500
X = DataFrame(weight=fill(1.0, N), x=fill([1.0], N));  # row = (weight=1.0, x=Vector{Float64})
y = vcat(10.0 .+ 2.0*randn(round(Int(N/3))), 17.0 .+ 2.0*randn(round(Int(N/3))), 24.0 .+ 2.0*randn(round(Int(N/3))));
#y = vcat(10.0 .+ 2.0*randn(round(Int(N/3))), 20.0 .+ 2.0*randn(round(Int(N/3))), 30.0 .+ 2.0*randn(round(Int(N/3))));
histogram(y, nbins=50)

# Sample-Fit-Combine
ensemble = MyEnsemble(NormalKDE(), 10, 0.8)  # Maximum of 10 components
lossfunc = (yhat, y) -> -logpdf(yhat, y)
clustercomponents!(ensemble, X, y, lossfunc)

# Results
println(ensemble.weights)
for mchn in ensemble.components
    println(predict(mchn, X[1:1, :x]))
end
println(loss(ensemble, X[!, :x], y, lossfunc))

