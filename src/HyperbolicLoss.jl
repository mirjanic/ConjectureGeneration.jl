using LossFunctions: SupervisedLoss

@doc raw"""
    HyperbolicLoss <: SupervisedLoss

TODO

"""
struct HyperbolicLoss <: SupervisedLoss end

(loss::HyperbolicLoss)(output::Number, target::Number) = 2 * atanh(
  min(0.99999,
      abs((output - target) / (output + target))
      )
)
