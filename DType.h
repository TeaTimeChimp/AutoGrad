#pragma once


// Define AUTOGRAD_DOUBLE (e.g. set CL=/DAUTOGRAD_DOUBLE) to build in double precision, which is useful for gradient checking.
#ifdef AUTOGRAD_DOUBLE
typedef double FP;
#else
typedef float FP;
#endif

// Device kernels are float only.
#if defined(AUTOGRAD_DOUBLE)&&defined(AUTOGRAD_CUDA)
#undef AUTOGRAD_CUDA
#endif
