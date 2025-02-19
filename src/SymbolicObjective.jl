using DynamicExpressions
using SymbolicRegression: Dataset
using LossFunctions: Loss, mean

"""

Contains auxiliary data needed to evaluate signum loss.

# Fields

- `requiredOps::AbstractVector{Tuple{Integer, Integer}}`: 
    List of pairs `(degree, index)` of operators to appear in solution.

- `complexityWeight::Real`:
    How much to punish _lower_ complexities to find expressions of all sizes.

"""
struct SymbolicObjective <: Function
  requiredOps::AbstractVector{Tuple{Integer,Integer}}
  loss_function::Loss
  complexityWeight::Real
  unusedFunctionPenalty::Real
end

function SymbolicObjective(requiredOps, loss_function; complexityWeight=0, unusedFunctionPenalty=1e5)
  return SymbolicObjective(requiredOps, loss_function, complexityWeight, unusedFunctionPenalty)
end


function (objective::SymbolicObjective)(tree, dataset::Dataset{T,L}, options, idx) where {T,L}
  X = copy(dataset.X)
  y = copy(dataset.y)
  if idx !== nothing  # Batching support
    X = X[:, idx]
    y = y[idx]
  end

  complexity = 0

  # Punish not using certain operators
  required_f = objective.requiredOps
  required_f = Dict{Tuple{Integer,Integer},Integer}([f => false for f in required_f])
  foreach(tree) do node
    complexity += 1
    if node.degree > 0 && node.feature == 0 && !(node.constant)
      required_f[(node.degree, node.op)] = true
    end
  end

  unusedFunctionCount = sum(.!values(required_f))
  if unusedFunctionCount > 0
    # Short circuit return if not all required functions are used
    return unusedFunctionCount * objective.unusedFunctionPenalty
  end

  # Evaluate expression tree
  y_pred, ok = eval_tree_array(tree, X, options)
  if !ok
    return L(Inf)
  end

  # Loss
  loss = mean(objective.loss_function, y_pred, y) + 1 / complexity * objective.complexityWeight
  return L(loss)
end
