#=
Bootstrp Aggregation (Bagging).

INPUT:
- ytrain, Xtrain. Training data.
- ytest, Xtest. Test data.
- model. An instance of MLJ.Supervised.
- sampling_fraction. Float64. sample size = n = N * sampling_fraction,  where N = length(ytrain).
=#

using MLJ
using SampleFitCombine


y = xxx
X = xxx


N = size(ytrain, 1)
n = Int(round(sampling_fraction * n))
K = 5
w = ProbabilityWeights([1.0/N for i = 1:N])
i_all   = collect(1:N);
i_train = fill(0, n);
mchn    = machine(model, Xtrain, ytrain)
for k = 1:K
    sample!(Random.GLOBAL_RNG, i_all, w, i_train; replace=true, ordered=true);  # ordered means sorted
    fit!(mchn, rows=i_train)
    combine!(ensemble, mchn, :uniform)
    stopcriteria(ensemble) && break
end
