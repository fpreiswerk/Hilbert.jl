module Hilbert

"""
    hilbert(x, n=Int64[])
Analytic signal, computed using the Hilbert transform.
`y = hilbert(x)` gives the analytic signal `y = x + i*xi` where `xi` is the
Hilbert transform of vector `x`. If `x` is complex, only the real part is used.
If `x` is a matrix, then `hilbert` operates along columns.

**Example**
```julia
x = randn(10,10)
y = hilbert(x)
```

`hilbert(x,n)` computes the n-point Hilbert transform.
"""
function hilbert(x, n=Int64[])

  if(!(eltype(x) <: Number))
    error("Only numerical input is supported")
  elseif(length(size(x))>2)
    error("Only vectors and matrices are supported")
  end

  x_ = copy(x)

  if(eltype(x_) <: Complex)
    warn("Using real part, ignoring complex part")
    x_ = real(x_)
  end

  # work along columns
  size(x_,1)==1 ? x_ = permutedims(x_,[2 1]) : nothing

  #isempty(n) ? n = size(x_,1) : nothing
  if(!isempty(n) && n<size(x_,1))
    x_ = x_[1:n,:]
  elseif(!isempty(n) && n>size(x_,1))
    x_ = cat(1,x_,zeros(n-size(x_,1),size(x_,2)))
  else
    n = size(x_,1)
  end

  xf = fft(x_,1)
  h = zeros(Int64,n)
  if n>0 && n % 2 == 0
    #even, nonempty
    h[1:div(n,2)+1] = 1
    h[2:div(n,2)] = 2
  elseif n>0
    #odd, nonempty
    h[1] = 1
    h[2:div(n + 1,2)] = 2
  end
  x_ = ifft(xf .* h[:,ones(Int64,size(xf,2))],1)

  # restore to original shape if necessary
  size(x,1)==1 ? x_ = permutedims(x_,[2 1]) : nothing

  return x_

end


end # module
