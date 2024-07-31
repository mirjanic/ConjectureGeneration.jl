
"""
Defines an interval domain `[lo,hi] ∩ T`, where `T<:Number` is a set of numbers (Z, R, ...)

# Fields

- `lo::T`: Interval lower bound
- `hi::T`: Interval upper bound
- `cast::Function`: Mapping from `U` to `T`

"""
struct Domain{T<:Number}
  lo::T
  hi::T
  cast::Function
end

"""
Wraps a function onto a domain.

# Arguments
- `cont::Function`: Operator from `U^n` to `U`
- `dom::Domain{T}`: Domain information struct

# Returns
A function from `T^n` to `T`.

"""
function wrap(cont::Function, dom::Domain{T}) where {T<:Number}
  # We need to give a name to the return function to be compatible with SymbolicRegression.jl
  # So, choose the name to be `nameof(cont) + "_" `
  # First we store this in return_func_name
  # Then we construct the function in an eval
  # TODO Is there a better way?
  return_func_name::Symbol = Symbol(string(nameof(cont)), "_")
  @eval function $(return_func_name)(x::Vararg{U})::U where {U<:Number}
    if any(x .> $(dom.hi)) || any(x .< $(dom.lo))
      return NaN
    end
    x = $(T).($(dom.cast).(x))
    $(cont)(x...)
  end
end



