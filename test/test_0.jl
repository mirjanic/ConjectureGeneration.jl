import SymbolicRegression: SRRegressor
import MLJ: machine, fit!, predict, report

using ConjectureGeneration

N = 100
X = (Vector{Float32}(rand(1:100, N)),)
y = zeros(Float32, N)

loss = SignumLoss([(1, 1), (2, 1), (2, 2)])  # cos, +, *

model = SRRegressor(
    niterations=20,
    binary_operators=[+, *, /],
    unary_operators=[cos, exp],
    progress=true,
    loss_function=loss,
    complexity_of_constants=100
)

mach = machine(model, X, y)
fit!(mach)

