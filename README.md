# SampleFitCombine.jl

## Ensembles

An ensemble is weighted combination of models.

It has the form:

```julia
struct Ensemble{T <: Supervised}
    components::Vector{machine{T}}
    weights::Vector{Float64}
end
```

To train an ensemble of `K` components, we follow this format:

```julia
for k = 1:K
    sample
    fit
    combine
    stopcriteria(ensemble) && break
end
```

Specifying the `sample`, `fit` and `combine` steps independently from one another
permits a wide range of ensembles to be constructed with a unified approach.

The function `stopcriteria(ensemble)` specifies the conditions under which the loop should stop appending components to the ensemble.
By default `stopcriteria(ensemble)=false`.
This allows ensembles to be trained without knowing the number of components in advance, while also providing safety from an infinite loop.

## Example: Bagging

```julia
```

See the `examples` directory for other more complex ensembles.

## Sample


## Fit


## Combine

The combine method assigns a weight to each component of the ensemble such that the weights sum to 1.
The 3 weighting schemes are:
1. :uniform. Assign equal weight to each component.
2. weights::Vector{Float64}. Assign the pre-specified weights to the components.
3. :optimize. Find weights that minimize...

## Predict

For an ensemble of classifiers, `predict(ensemble, Xnew)` outputs the category with the highest total weight among the predictions of the ensemble's components.
For example, if an ensemble of 3 classifiers has weight vector `[0.1, 0.6, 0.3]`,
and the 3 components `predict` categories `a`, `b` and `a` respectively, then the ensemble's prediction will be `b` because it has a weight of 0.6,
whereas `a` has a weight of 0.4.

For all other model types `predict(ensemble, Xnew)` is a weighted average of the component predictions.
