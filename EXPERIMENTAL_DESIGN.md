# EXPERIMENTAL_DESIGN.md — Model Extraction Benchmark (v1.1)

## 1. Goal
To rigorously benchmark the performance of various **Model Extraction Attacks** (High Fidelity, Query Efficiency) under controlled experimental conditions, varying:
- **Victim Models**: Architecture & Training Data Domain
- **Substitute Models**: Architecture
- **Auxiliary Data**: Seed & Surrogate Datasets

## 2. Experimental Matrix

### A. Victim Models (Target)
| Domain | Input Shape | Architecture | Dataset | Checkpoint Strategy |
|--------|-------------|--------------|---------|---------------------|
| **MNIST** | 1x28x28 | LeNet | MNIST | Pre-trained (High Acc) |
| **CIFAR10** | 3x32x32 | ResNet18 | CIFAR10 | Pre-trained (High Acc) |

### B. Substitute Models (Attacker)
The attacker initializes these architectures from scratch.
| Architecture | Input Support | Notes |
|--------------|---------------|-------|
| **LeNet** | 1x28x28, 3x32x32 | Standard lightweight CNN |
| **ResNet18** | 1x28x28, 3x32x32 | Standard residual network |
| **MobileNetV2** | 1x28x28, 3x32x32 | Efficient architecture (optional) |

**Requirement**: Model factory must support dynamic `input_channels` (1 or 3) and `input_size` (28 or 32) for ALL architectures.

### C. Datasets (Auxiliary & Evaluation)
| Role | Datasets | Usage |
|------|----------|-------|
| **Problem Domain (Test)** | MNIST, CIFAR10 | Evaluation (Agreement/Accuracy) |
| **Seed Data** | EMNIST, FashionMNIST, CIFAR100 | Initial query pool / Partial data |
| **Surrogate Data (Pool)** | EMNIST, FashionMNIST, SVHN, GTSRB, CIFAR100 | Unlabeled query pool for attacks |

**Requirement**: Loaders must support `SeedDataset` and `SurrogateDataset` interfaces for all above datasets.

### D. Attacks (Benchmark Targets)
| Attack | Type | Key Parameter | Output Mode |
|--------|------|---------------|-------------|
| **Random** | Baseline | - | Soft/Hard |
| **ActiveThief** | Pool-based | Strategy (Entropy, K-Center, DFAL) | Soft |
| **SwiftThief** | Pool-based | Contrastive Loss | Soft |
| **CloudLeak** | Pool-based | Adversarial Examples | Soft |
| **CopycatCNN** | Pool-based | Fake Dataset (NPD) | Soft |
| **Knockoff Nets** | Pool-based | Random/Adaptive Selection | Soft |
| **InverseNet** | Hybrid | Inversion -> Pool | Soft |
| **Black-box Dissector** | Pool-based | Hard Label (CAM) | Hard |
| **Black-box Ripper** | Pool-based | Hard Label (GAN) | Hard |
| **DFME** | Data-free | GAN (L1 Loss) | Soft |
| **MAZE** | Data-free | GAN (Gradient Est) | Soft |
| **DFMS-HL** | Data-free | GAN (Diversity) | Soft |
| **GAME** | Data-free | GAN (Component-based) | Soft |
| **ES Attack** | Data-free | Synthesis (DNN/OPT) | Soft |

## 3. Pipeline Requirements & Gap Analysis

### Current Gaps (to be fixed immediately)
1. **Dynamic Input Support**:
   - `mebench/models/substitute_factory.py` currently hardcodes LeNet to 32x32 (fixed in `lenet_mnist` but needs general solution).
   - ResNet/MobileNet must handle 1-channel input (conv1 modification).
2. **Dataset Loaders**:
   - `GTSRB` support missing in `mebench/data/loaders.py`.
   - `SeedDataset` currently only supports MNIST/CIFAR10. Needs expansion to EMNIST/FashionMNIST.
3. **Attack Implementation Hardcoding**:
   - Many attacks (ActiveThief, Dissector, etc.) have `input_channels=3` hardcoded.
   - Must use `state.metadata["input_shape"]` to determine channels dynamically.
4. **Engine**:
   - `input_shape` validation must align with Victim config.

## 4. Execution Plan
1. **Refactor Model Factory**: Implement generic `create_substitute(arch, input_channels, num_classes)` that adapts internal layers (conv1, fc) automatically.
2. **Expand Loaders**: Implement GTSRB loader and generic Seed/Surrogate wrappers.
3. **Fix Attacks**: Remove `input_channels=3` hardcoding; use state metadata.
4. **Smoke Test**: Run a matrix of (Victim x Substitute x Dataset) to verify shapes and forward passes.
