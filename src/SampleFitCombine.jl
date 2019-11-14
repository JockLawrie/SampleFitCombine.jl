module SampleFitCombine

export Ensemble,
       ProbabilityWeights,
       sample!, fit!, combine!, predict

using Distributions  # MixtureModel
using MLJ
using MLJBase    # target_scitype
using Random     # GLOBAL_RNG
using StatsBase  # sample!

#import StatsBase.sample!  # Re-exported
#import MLJ.fit!           # Re-exported
import MLJ.predict         # To extend with SampleFitCombine.predict


struct Ensemble{T <: Supervised}
    components::Vector{Machine{T}}
    weights::Vector{Float64}
end

Ensemble(T) = Ensemble(T[], Float64[])


combine!(x) = nothing


predict(ensemble::Ensemble{<:Probabilistic}, Xnew) = MixtureModel([predict(mchn, Xnew) for mchn in ensemble.components], copy(ensemble.weights))

function predict(ensemble::Ensemble, Xnew)
    MLJBase.target_scitype(Type{T}) == AbstractVector{<:Finite} && return predict_classifier(ensemble, Xnew)
    result     = 0.0
    components = ensemble.components
    weights    = ensemble.weights
    for (k, component) in enumerate(ensemble.components)
        result += weights[k] * predict(component, Xnew)
    end
    result
end

#losses = [loss(yhati, yi) for (yhati, yi) in zip(yhat, y)]
#w = ProbabilityWeights(pw; wsum=sum(losses))


end
