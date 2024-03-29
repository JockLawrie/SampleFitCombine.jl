module SampleFitCombine

export Ensemble,  # Type
       components, ncomponents, max_ncomponents, weights,  # Accessor functions
       fitcomponent!,
       combine_uniform_weights!, combine_prespecified_weights!, combine_optimal_weights!,
       predict, loss

using Distributions  # MixtureModel
using Logging
using MLJ
using MLJBase
using Optim
using Tables

# Methods that this package extends
import Distributions.components
import Distributions.ncomponents
import MLJ.predict

################################################################################
# Type

"""
Ensemble{M<:Model}
  model::M
  components::Vector{Machine{M}}  # Trained components
  weights::Vector{Float64}        # Mixing weights
  Kmax::Int                       # Maximum number of components
"""
struct Ensemble{M<:Model}
    model::M
    components::Vector{Machine{M}}
    weights::Vector{Float64}
    Kmax::Int

    function Ensemble(model, components, weights, Kmax)
        @assert Kmax > 0
        new{typeof(model)}(model, components, weights, Kmax)
    end
end

Ensemble(model, Kmax) = Ensemble(model, Machine{typeof(model)}[], Float64[], Kmax)

################################################################################
# Accessor functions

"Returns the vector of trained ensemble components."
components(ensemble::Ensemble) = ensemble.components

"Returns the number of trained components."
ncomponents(ensemble::Ensemble) = length(ensemble.components)

"Returns the maxmimum number of components allowed."
max_ncomponents(ensemble::Ensemble) = ensemble.Kmax

"Returns the ensemble's mixing weights."
weights(ensemble::Ensemble) = ensemble.weights

################################################################################
# Fit

function fitcomponent!(ensemble::Ensemble, X, y; rows=collect(1:size(y, 1)))
    ncomponents(ensemble) >= max_ncomponents(ensemble) && error("Ensemble already has K=$(max_ncomponents(ensemble)) trained components")
    mchn = machine(ensemble.model, X, y)
    fit!(mchn, rows=rows)
    push!(components(ensemble), mchn)
end

################################################################################
# Combine

function combine_uniform_weights!(ensemble)
    empty!(weights(ensemble))
    nk = ncomponents(ensemble)
    w  = 1.0 / nk
    for k = 1:nk
        push!(weights(ensemble), w)
    end
    nothing
end

function combine_prespecified_weights!(ensemble, w)
    # Checks
    nk  = ncomponents(ensemble)
    nk == 0 && @warn "Ensemble has no components to combine." && return
    nw  = length(w)
    nw != nk && error("$(nw) weights were supplied to combine!, expecting $(nk) weights, 1 for each component.")

    # Ensure that ensemble.weights has nk elements
    wt = weights(ensemble)
    nw = length(wt)
    if nw < nk
        for k = (nw + 1):nk
            push!(wt, 0.0)
        end
    elseif nw > nk
        for k = 1:(nw - nk)
            pop!(wt)
        end
    end

    # Replace existing weights with new weights
    for k = 1:nk
        @inbounds wt[k] = w[k]
    end
    nothing
end

function combine_optimal_weights!(ensemble, X, y, lossfunc)
    logits = fill(0.0, ncomponents(ensemble) - 1)
    w      = fill(0.0, ncomponents(ensemble))
    pred_components = construct_prediction_components(ensemble, X)
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

function computeweights!(result, logits)
    wtotal    = 1.0
    result[1] = 1.0  # exp(0.0)
    K = length(result)
    for k = 2:K
        @inbounds w = exp(logits[k-1])
        @inbounds result[k] = w
        wtotal += w
    end
    result ./= wtotal
end

################################################################################
# Predict

function predict(ensemble::Ensemble, X)
    # Init result
    n = size(X, 1)
    pred_components = construct_prediction_components(ensemble, X)
    pred = nothing
    for row in Tables.rows(X)
        pred = predictrow!(ensemble, row, pred_components)
        !ismissing(pred) && break
    end
    result = Vector{Union{Missing, typeof(pred)}}(undef, n)

    # Populate result
    i = 0
    result[1] = pred
    for row in Tables.rows(X)
        i += 1
        i == 1 && continue  # Already calculated result[1]
        result[i] = predictrow!(ensemble, row, pred_components)
    end
    result
end


function predictrow!(ensemble::Ensemble, row, pred_components)
    for (k, mchn) in enumerate(components(ensemble))
        pred = predict(mchn, row)
        ismissing(pred) && return missing
        pred_components[k] = pred
    end
    MixtureModel(pred_components, weights(ensemble))
end


function construct_prediction_components(ensemble, X)
    for row in Tables.rows(X)
        pred = predict(components(ensemble)[1], row)
        ismissing(pred) && continue
        return Vector{typeof(pred)}(undef, ncomponents(ensemble))
    end
end

################################################################################
# Loss

"Returns: lossfunc(predict(ensemble, X), y)"
function loss(ensemble, X, y, lossfunc)
    size(X, 1) != size(y, 1) && error("X and y do not have the same number of observations.")
    pred_components = construct_prediction_components(ensemble, X)
    loss!(pred_components, ensemble, X, y, lossfunc)
end

function loss!(pred_components, ensemble, X, y, lossfunc)
    result = 0.0
    i = 0
    for row in Tables.rows(X)
        i      += 1
        yhat    = predictrow!(ensemble, row, pred_components)
        ismissing(yhat) && continue
        result += lossfunc(yhat, y[i])
    end
    result / i
end

end
