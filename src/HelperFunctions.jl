module HelperFunctions

using PyCall
using Combinatorics
using Primes

function eulerTotient(n::Integer)::Integer
  result = n
  p = 2
  while p * p <= n
    if n % p == 0
      while n % p == 0
        n = div(n, p)
      end
      result -= div(result, p)
    end
    p += 1
  end
  if n > 1
    result -= div(result, n)
  end
  return result
end

op(x, y) = y * sign(x)

prime(n::Integer)::Integer = pyimport("sympy").ntheory.generate.prime(n)

# TODO this call has been depraceated and moved to 
# sympy.functions.combinatorial.numbers.primeomega
primeOmega(n::Integer)::Integer = pyimport("sympy").ntheory.factor_.primeomega(n)

primorial(n) = Combinatorics.primorial(n)

eulerSigma(n::Integer)::Integer = sum(Primes.divisors(n))

harmonic(x::Integer)::Real = sum(1 ./ range(1, x))

safelog(x) = x > 1 ? log(x) : convert(typeof(x), 0)

end
