using LossFunctions: SupervisedLoss

@doc raw"""
    HyperbolicLoss <: SupervisedLoss

TODO

"""
struct HyperbolicLoss <: SupervisedLoss end

(loss::HyperbolicLoss)(output::Number, target::Number) = atanh( 
  min(0.9,
      abs((output - target) / (output + target))
      )
)

# (loss::HyperbolicLoss)(output::Number, target::Number) = abs(log(abs(output / target)))
