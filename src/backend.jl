abstract type ComputeBackend end

struct CPUBackend <: ComputeBackend end
struct CUDABackend <: ComputeBackend end

function has_cuda()
   try 
       @eval using CUDA
       return CUDA.functional()
   catch
       return false
   end
end
