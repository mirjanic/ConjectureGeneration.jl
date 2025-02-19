import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report
import Combinatorics
import Primes

include("ConjectureGeneration.jl")
using .ConjectureGeneration: SymbolicObjective, SignumLoss, Domain, wrap, HyperbolicLoss, HelperFunctions as FnBox


N = 100
X = (x=Vector{Float32}(rand(1:100, N)),)
# y = zeros(Float32, N)

target_func(x) = exp(Base.MathConstants.γ) * x * log(log(x)) - sum(Primes.divisors(x))
y = target_func.(ceil.(Int32, X.x))

# loss = SymbolicObjective([(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)], SignumLoss())
loss = SymbolicObjective([], HyperbolicLoss())
dom = Domain{Int32}(1, 100, ceil)


model = SRRegressor(
  populations=32,
  # niterations=1000,
  ncycles_per_iteration=1000,
  # warmup_maxsize_by=0.2,
  # maxsize=20,
  binary_operators=[+, *
  # binary_operators=[FnBox.op, +, *, /,
  ],  # <---- functions go here
  unary_operators=[exp, wrap(FnBox.eulerTotient, dom)
  # unary_operators=[wrap(Combinatorics.primorial, dom), wrap(FnBox.eulerTotient, dom), wrap(FnBox.primeOmega, dom), cos, exp
  ],  # <---- or here
  complexity_of_constants=1000,
  # should_simplify=true,
  # progress=true,
  loss_function=loss,
  # parsimony=1e-2,
)
mach = machine(model, X, y)

fit!(mach)

# report(mach)
