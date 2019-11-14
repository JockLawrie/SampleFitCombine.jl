#=
Bootstrp Aggregation (Bagging).

INPUT:
- ytrain, Xtrain. Training data.
- ytest, Xtest. Test data.
- model. An instance of MLJ.Supervised.
- sampling_fraction. Float64. sample size = n = N * sampling_fraction,  where N = length(ytrain).
=#

using MLJ
import GLM
using MLJModels.GLM_
using SampleFitCombine

# Data
N = 1000
x = 5.0 .+ 10.0 * rand(N)
y = 2.0 .+ 3.0 .* x .+ 2.0 .* randn(N)  # Y ~ N(2 + 3x, 2^2), where x is Uniform(5, 15)

# Model
model = @load MLJModels.GLM_.LinearRegressor

# Hyperparameters
sampling_fraction = 0.8

# sample-fit-combine
n = Int(round(sampling_fraction * N))
K = 5
w = ProbabilityWeights([1.0/N for i = 1:N])
i_all   = collect(1:N)
i_train = fill(0, n)
mchn    = machine(model, x, y)
for k = 1:K
    sample!(Random.GLOBAL_RNG, i_all, w, i_train; replace=true, ordered=true);  # ordered means sorted
    fit!(mchn, rows=i_train)
    combine!(ensemble, mchn, :uniform)
    stopcriteria(ensemble) && break
end
