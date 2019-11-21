using Test
using SampleFitCombine

using Clustering
using Distributions  # Normal(m, s)
using Logging
using MLJ
using MLJBase
using StatsBase  # wsample!, ProbabilityWeights
using UnicodePlots

import SampleFitCombine.weights  # Because StatsBase also exports weights, so we need to specify which method we want

################################################################################
# Model

"Normal density estimate with no predictor variables."
mutable struct NormalDensity <: MLJBase.Probabilistic
end

MLJBase.target_scitype(::Type{NormalDensity}) = AbstractVector{<:MLJ.Continuous}

function MLJBase.fit(model::NormalDensity, verbosity::Integer, X, y)
    n = length(y)
    W = 0.0  # running total weight
    A = 0.0  # running mean
    Q = 0.0  # running sum of squares (the numerator of the variance)
    for i = 1:n
        w  = 1.0  # TODO: Allow other weights to be passed in
        w == 0.0 && continue
        yi = y[i]
        W += w
        Aprev = A
        A += w * (yi - A) / W
        Q += w * (yi - Aprev) * (yi - A)
    end
    [A, sqrt(Q/W)], nothing, nothing  # fitresult, cache, report
end

MLJBase.predict(model::NormalDensity, fitresult, Xnew) = Normal(fitresult[1], fitresult[2])

################################################################################
# Sample-Fit-Combine: Cluster components

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
    # Fit 1 cluster
    fitcomponent!(ensemble, X, y)
    combine_uniform_weights!(ensemble)     # Give the component a weight of 1

    # Fit k clusters, k = 2:K
    Lmin  = loss(ensemble, X, y, lossfunc)  # Minimum loss achieved so far
    N     = length(y)
    Kmax  = max_ncomponents(ensemble)
    probs = fill(0.0, 1, N)
    for k = 2:Kmax
        pred_components = SampleFitCombine.construct_prediction_components(ensemble, X[1])
        for i = 1:N
            pred = SampleFitCombine.predict!(ensemble, X[i], pred_components)
            probs[1, i] = cdf(pred, y[i])
        end
        res = kmeans(probs, k)
        a   = assignments(res)
        old_components = copy(components(ensemble))
        old_weights    = copy(weights(ensemble))
        for k2 = 1:(k-1)
            popfirst!(components(ensemble))
        end
        for cluster = 1:k
            # Sample
            sample = a .== cluster
            ytrain = y[sample]
            Xtrain = X[sample]

            # Fit
            fitcomponent!(ensemble, Xtrain, ytrain)
        end
        combine_optimal_weights!(ensemble, X, y, lossfunc)  # Combine

        # Stopping criteria
        L = loss(ensemble, X, y, lossfunc)
        @info "K = $(k). Loss = $(L). Lmin = $(Lmin)"
        if L > Lmin || abs((L - Lmin)/Lmin) <= 0.005
            for k2 = 1:k
                popfirst!(components(ensemble))
            end
            for k2 = 1:length(old_components)
                push!(components(ensemble), old_components[k2])
            end
            combine_prespecified_weights!(ensemble, old_weights)
            break
        end
        Lmin = L
        ncomponents(ensemble) == max_ncomponents(ensemble) && break
    end
    Lmin
end

################################################################################
println("")
@info "Example 1: Data: 2 Normal components. Sample: simple random sample. combine: uniform_weights."
@info "    Also known as Bagging"

# Data: Y ~ 0.5*N(10, 2^2) + 0.5*N(20, 2^2)
N = 1000
X = fill([1.0], N)
y = vcat(10.0 .+ 2.0*randn(Int(0.5*N)), 20.0 .+ 2.0*randn(Int(0.5*N)))
#histogram(y, nbins=30)

# Sample-Fit-Combine
ensemble = Ensemble(NormalDensity(), 10)  # Maximum of components
samplingfraction = 0.8
n        = Int(round(samplingfraction * N))         # Sample size
w        = ProbabilityWeights([1.0/N for i = 1:N])  # Sample weights
i_all    = collect(1:N)  # Population row indices
i_train  = fill(0, n)    # Sample row indices
for k = 1:max_ncomponents(ensemble)
    wsample!(i_all, w, i_train; replace=true, ordered=false);  # Sample
    fitcomponent!(ensemble, X, y; rows=i_train)  # Fit
end
combine_uniform_weights!(ensemble)  # Combine
for machine in components(ensemble)
    println(predict(machine, X[1]))
end
println(weights(ensemble))

# loss
lossfunc = (yhat, y) -> -logpdf(yhat, y)
loss_bagging = loss(ensemble, X, y, lossfunc)

# Predict
Xtest = X[1]
ytest = y[1]
println(predict(ensemble, Xtest))

# Alternative combine: Combine components with pre-specified weights
w   = rand(max_ncomponents(ensemble))
w ./= sum(w)
combine_prespecified_weights!(ensemble, w)
println(weights(ensemble))
println(loss(ensemble, X, y, lossfunc))

# Alternative combine: Combine components with optimized weights
combine_optimal_weights!(ensemble, X, y, lossfunc)
println(weights(ensemble))
println(loss(ensemble, X, y, lossfunc))

################################################################################
println("")
@info "Example 2: Data: unchanged. Sample-Fit-Combine: clustercomponents."

# Sample-Fit-Combine
ensemble = Ensemble(NormalDensity(), 10)  # Maximum of 10 components
lossfunc = (yhat, y) -> -logpdf(yhat, y)
clustercomponents!(ensemble, X, y, lossfunc)
for machine in components(ensemble)
    println(predict(machine, X[1]))
end
println(weights(ensemble))

# loss
loss_cc = loss(ensemble, X, y, lossfunc)
@info "Bagging: loss = $(loss_bagging);  Cluster components: loss = $(loss_cc)"

################################################################################
println("")
@info "Example 3: Data: 3 components. Sample-Fit-Combine: clustercomponents."

# Data: Y ~ 0.333*N(10, 2^2) + 0.333*N(17, 2^2) + 0.333*N(24, 2^2)
N = 1500
X = fill([1.0], N)
y = vcat(10.0 .+ 2.0*randn(round(Int(N/3))), 17.0 .+ 2.0*randn(round(Int(N/3))), 24.0 .+ 2.0*randn(round(Int(N/3))));
#histogram(y, nbins=50)

# Sample-Fit-Combine
ensemble = Ensemble(NormalDensity(), 10)  # Maximum of 10 components
lossfunc = (yhat, y) -> -logpdf(yhat, y)
clustercomponents!(ensemble, X, y, lossfunc)
for machine in components(ensemble)
    println(predict(machine, X[1]))
end
println(weights(ensemble))

# loss
println(loss(ensemble, X, y, lossfunc))
