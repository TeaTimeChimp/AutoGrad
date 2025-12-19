# AutoGrad

A small C++ automatic differentiation engine with NumPy-style ND arrays, a minimal neural network API, and optional CUDA-accelerated matrix multiplication. It ships with basic layers, losses, and optimisers plus a set of self-checking tests.

## Features
- Tensor/NDArray core (`Tensor.h`, `NDArray.h`): broadcasting, slicing, dot/transpose, softmax/log-softmax, var/mean/sum, masking, reshape/unsqueeze/squeeze, dropout, argmax, cat/stack, gather/index_select, and random initialisers.
- Autograd: dynamic computation graph with per-op gradient kernels (`Ko*.cpp/.h`) and safety checks for gradient completeness.
- Layers (`Layer.h`, `Sequential.h`): Linear (He init, optional bias/activation), Embedding, Tanh/ReLU activations, Dropout.
- Losses: MSE (`MSELoss.h`) and Cross Entropy (`CrossEntropyLoss.h`).
- Optimisers: SGD and Adam (`SGD.h`, `ADAM.h`), including momentum buffers on parameters.
- Utilities: ND thread pool (`NDThreadPool.h`), tiled matmul and CUDA kernel (`CUDA/matrixMul.cu`), file I/O for tensors, and a battery of unit-style tests (`Test_*.cpp`).

## Repo layout
- `AutoGrad.cpp` - sample entry point running tests and small training demos.
- `Tensor.h`, `NDArray.h`, `NDData.h*` - tensor implementation, storage, and math kernels.
- `Ko*.{h,cpp}` - forward/backward definitions for each tensor op.
- `Layer.h`, `Sequential.h`, `Linear.h`, `Embedding.h`, `Relu.h`, `Tanh.h`, `Dropout.h` - model building blocks.
- `CrossEntropyLoss.h`, `MSELoss.h` - loss functions.
- `SGD.h`, `ADAM.h`, `RMSProp.h`, `Optimiser.h` - optimisation steps.
- `Test*.{h,cpp}` - coverage for broadcasting, NDArray ops, Tensor autograd, Linear/Embedding layers, distributions, thread pool, etc.
- `CUDA/` - CUDA helpers and a matrix multiply kernel (optional).

## Build
- Open `AutoGrad.sln` in Visual Studio 2022 (x64). The project targets C++20 and uses PPL; CUDA support requires the CUDA toolkit if you want GPU matmul.
- Or from a VS Developer PowerShell prompt:
  ```powershell
  msbuild AutoGrad.sln /p:Configuration=Release /p:Platform=x64
  ```

## Quickstart
`AutoGrad.cpp` already wires up quick demos:
- An MSE example with scalar tensors and SGD.
- A toy "x*y=10" optimisation loop.
- A tiny embedding -> tanh -> linear classifier trained with cross entropy.

Minimal sketch:
```cpp
#include "Tensor.h"
#include "Sequential.h"
#include "Embedding.h"
#include "Tanh.h"
#include "Linear.h"
#include "CrossEntropyLoss.h"
#include "SGD.h"

int main() {
    TensorPtr data   = Tensor::New(NDData::New({4}, {1,2,1,2}), true);
    TensorPtr target = Tensor::New(NDData::New({4}, {0,1,0,1}), true);

    Sequential model({
        Embedding::New(3, 3),
        Tanh::New(),
        Linear::New(3, 4, "clf", true)
    });

    CrossEntropyLoss loss;
    SGD optim(model.GetParameters(), 0.1f);

    for (int i = 0; i < 10; ++i) {
        TensorPtr pred  = model.Forward(data);
        TensorPtr value = loss.Forward(pred, target);
        value->Backward();
        optim.Step();
    }
}
```

## Tests
Running the `AutoGrad` executable triggers `Test()` in `Test.cpp`, which exercises NDArray math, broadcasting, Tensor autograd, Linear/Embedding layers, distributions, and the thread pool. Failures throw `TestFailed`.

## Notes
- Parameters carry momentum buffers to support optimisers; gradient buffers are zeroed after each `Step` by default.
- Softmax/log-softmax are numerically stable (max subtraction); cross-entropy expects class indices (one per batch item).
- Tensor save/load uses raw file I/O (`NDArray::Save/Load`) for quick serialization; shapes are persisted alongside data.
