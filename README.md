# SampleFitCombine.jl

## Ensembles

An `Ensemble` is weighted combination of models.

It is defined as follows:

```julia
struct Ensemble{M<:Model}
    model::M
    components::Vector{Machine{M}}  # Trained components
    weights::Vector{Float64}        # Mixing weights
    Kmax::Int                       # Maximum number of components
end
```

## Sample-Fit-Combine

To train an ensemble of `K` components, we follow the sample-fit-combine strategy as described below.
This permits a wide range of methods for constructing ensembles.

1. We start with an `Ensemble` with 0 components, typically initialised with `ensemble = Ensemble(MyModel(), Kmax)`,
   for some model type `MyModel`.
   Here `Kmax` is the maximum number of components allowed in the ensemble
   (the actual number may or may not be known in advance).

2. __Sample__: We take a sample of the training data. Example methods include:

  - Take a random sample (used in bagging).
  - Take a sample based on the loss of a previous fit.
  - Take the whole training set.

3. __Fit__: A component (machine) is trained on the sample and appended to the ensemble.
   In future we could train on the whole training set with weights applied.
   Indeed the sample methods listed above are special cases of this approach.

4. __Combine__: The package provides 3 ways to combine the components into an ensemble:

    a. `combine_uniform_weights!(ensemble)`: Set the same weight for each component.
    b. `combine_prespecified_weights!(ensemble, w)`: Set the weights to the pre-specified vector `w`.
    c. `combine_optimal_weights!(ensemble, X, y, lossfunc)`: Find the weights that minimize `lossfunc(ensemble, X, y, lossfunc)`

5. Repeat steps 2-4 until some stopping criterion is met.
   For example:
   - Stop if the ensemble has `Kmax` components.
   - Stop if `loss(ensemble, X, y, lossfunc)` hasn't changed much.

## Examples

See the test cases for examples of some of the possibilities, which include bagging and clustering.
