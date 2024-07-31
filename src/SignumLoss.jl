using LossFunctions: DistanceLoss

@doc raw"""
    signumLoss <: DistanceLoss

TODO

```math
L(r) = \begin{cases} 1 & \quad \text{if } r < 0 \\ 0 & \quad \text{if } r \ge 0 \\ \end{cases}
```
"""
struct SignumLoss <: DistanceLoss end

(loss::SignumLoss)(difference::Number) = difference < 0
