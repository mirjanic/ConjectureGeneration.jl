# import SymbolicRegression: SRRegressor, @template_spec
using SymbolicRegression
using SymbolicRegression.ComposableExpressionModule: apply_operator
using SymbolicRegression.TemplateExpressionMacroModule: template_spec
import MLJ: machine, fit!, predict, report
import Combinatorics
import Primes

include("ConjectureGeneration.jl")
using .ConjectureGeneration: SymbolicObjective, SignumLoss, Domain, wrap, HyperbolicLoss, HelperFunctions as FnBox


N = 100
x = Vector{Int64}(rand(2:100, N))

Xdata = (x=Float64.(x),)

target_func(x) = exp(Base.MathConstants.γ) * x * log(log(x)) - FnBox.eulerSigma(x)
y = target_func.(x)

loss = SymbolicObjective(HyperbolicLoss())
dom = Domain{Int32}(1, 1000, ceil)

# Define Expression Constraint
template =  @template_spec(expressions=(f, g)) do x
  f(x) + g(x)
end


model = SRRegressor(
  populations=32,
  niterations=1000,
  ncycles_per_iteration=1000,
  # warmup_maxsize_by=0.2,
  # maxsize=20,
  binary_operators=[+, *,
  # binary_operators=[FnBox.op, +, *, /,
  ],  # <---- functions go here
  unary_operators=[exp, FnBox.safelog, wrap(FnBox.eulerSigma, dom)
  # unary_operators=[wrap(Combinatorics.primorial, dom), wrap(FnBox.eulerTotient, dom), wrap(FnBox.primeOmega, dom), cos, exp
  ],  # <---- or here
  complexity_of_constants=4,
  # should_simplify=true,
  # progress=true,
  loss_function=loss,
  # parsimony=1e-2,
  expression_spec=template
)
mach = machine(model, Xdata, y)

fit!(mach)

# report(mach)
