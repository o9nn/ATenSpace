# ATenTorch Subcomponents

This directory contains specialized agent definitions for ATenTorch tensor operations and infrastructure.

## Agents

### ATenTorch-SYS
**System Operations Agent** - Manages tensor system-level operations, device allocation (CPU/GPU), memory management, and resource monitoring.

### ATenTorch-TH
**Torch Core Agent** - Implements fundamental tensor operations, linear algebra, and mathematical functions that form the foundation of all tensor computations.

### ATenTorch-THNN
**Torch Neural Network Primitives Agent** - Provides core neural network operations including convolutions, pooling, activation functions, normalization, and loss functions.

### ATenTorch-Optim
**Optimization Algorithms Agent** - Implements gradient-based optimization algorithms, learning rate schedulers, and parameter update strategies for training.

### ATenTorch-Graph
**Computational Graph Agent** - Manages automatic differentiation (autograd), gradient computation, and computational graph optimization for efficient backpropagation.

## Integration

These agents provide the tensor computation infrastructure:
- **SYS** manages hardware resources (CPU/GPU)
- **TH** provides core mathematical operations
- **THNN** implements neural network primitives
- **Optim** enables gradient-based learning
- **Graph** handles automatic differentiation

Together, they form a complete tensor computation framework that powers all neural and tensor-based operations in ATenCog.
