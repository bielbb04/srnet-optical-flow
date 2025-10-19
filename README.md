# Optical Flow Estimation with Deep Unfolding

A PyTorch implementation of optical flow estimation using deep unfolding techniques with residual neural networks. This project combines traditional variational methods with deep learning to estimate optical flow from image translations.

## Overview

This project implements a neural network-based approach to optical flow estimation that uses:
- **Deep Unfolding**: Unrolls an iterative optimization algorithm into a neural network
- **Residual Learning**: Uses ResNet-style blocks for flow refinement
- **Variational Framework**: Based on warping and gradient-based optimization

The model learns to estimate displacement fields (optical flow) between image pairs by minimizing a variational energy functional.

## Features

- Custom dataset loader for image patches with random translations
- ResNet-based architecture for flow estimation
- Warping operations with differentiable grid sampling
- Training with deep unfolding optimization
- TensorBoard logging for monitoring training progress
- Checkpoint saving and model evaluation

## Requirements

```bash
torch
torchvision
einops
numpy
matplotlib
scikit-image
imageio
tqdm
tensorboard
Pillow
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Project Structure

```
.
├── dataloader.py          # Dataset class for patch-based training
├── xarxa.py              # Neural network architecture (SRNet, ResBlock)
├── utils.py              # Utility functions (warping, gradients, etc.)
├── logger.py             # TensorBoard logging and checkpoint management
├── imports.py            # Common imports
├── proposta_per_translacions.py  # Main training script
└── README.md
```

## Model Architecture

### SRNet (Super-Resolution Network)
The core model consists of:
- Initial convolution layer to expand channels to feature space
- Multiple residual blocks (ResBlock) for feature processing
- Final convolution to produce 2-channel flow output

### ResBlock
Standard residual block with:
- Two convolutional layers with ReLU activation
- Skip connection for residual learning

## Dataset

The `Patches_translation` dataset:
- Loads images from a specified directory
- Generates random translations (shifts) for each image
- Creates image patches for training
- Supports configurable patch sizes (must be divisor of 200)
- Generates multiple training pairs per image (16 pairs by default)

### Data Generation
For each image:
1. Read original image (ground truth)
2. Apply random translation (0-5 pixels in x and y)
3. Extract patches if patch_size is specified
4. Generate flow ground truth

## Training

### Basic Usage

```python
python proposta_per_translacions.py --sampling 2 --dataset_path /path/to/data
```

### Arguments

- `--sampling`: Sampling factor (2, 4, or 8) - required
- `--dataset_path`: Path to dataset directory - required
- `--kernel_size`: Kernel size for convolutions (default: 3)
- `--features`: Number of feature channels (default: 128)
- `--blocks`: Number of residual blocks (default: 3)
- `--batch_size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 100)
- `--device`: Device for training (default: "cuda:0")
- `--patch_size`: Size of image patches (default: 64)
- `--nickname`: Optional nickname for experiment tracking

### Training Process

The training loop implements a deep unfolding approach:

1. **Outer Loop (Warping Iterations)**: 
   - Warp the shifted image using current flow estimate
   - Compute image gradients at warped locations

2. **Inner Loop (Proximal Iterations)**:
   - Compute residual ρ and its derivative
   - Update flow estimate: `x = u_k + λ * η * ρ * ∇ρ`
   - Apply neural network: `u_k = model(x)`

3. **Loss Computation**:
   - MSE loss between predicted flow and ground truth
   - Backpropagation and parameter updates

### Hyperparameters

- `iter_warp`: Number of warping iterations (default: 5)
- `iter_prox`: Number of proximal iterations (default: 15)
- `landa`: Lambda parameter for optimization (default: 0.5)
- `eta`: Eta parameter for step size (default: 0.25)

## Monitoring

Training progress is logged to TensorBoard:
- Training loss per epoch
- Validation loss per epoch
- Loss comparison plots
- Flow visualizations

View logs:
```bash
tensorboard --logdir=checkpoints/
```

## Checkpoints

Model checkpoints are automatically saved:
- `last.ckpt`: Latest model state
- `best.ckpt`: Best model based on validation loss

Checkpoints contain:
- Epoch number
- Model state dictionary

## Evaluation

Load a trained model:

```python
model = SRNet(sampling=2, kernel_size=3, features=128, blocks=3)
checkpoint = torch.load('path/to/checkpoint.ckpt')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Key Functions

### Image Warping
```python
warp_image(img, flow, padding_mode="border", interpolation="bilinear")
```
Warps an image according to optical flow using grid sampling.

### Gradient Computation
```python
gradient(u, stack=False)
```
Computes spatial gradients using finite differences.

### Variational Functions
- `rho()`: Computes the data term residual
- `rho_derivative()`: Computes residual and its gradient
- `warp_and_derivative()`: Warps image and computes warped gradients

## Results Visualization

The training script includes visualization of:
- Predicted flow vs ground truth flow
- Flow fields represented as color-coded images
- Training loss curves over epochs

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{optical_flow_unfolding,
  title={Optical Flow Estimation with Deep Unfolding},
  author={Your Name},
  year={2025}
}
```

## License

[MIT License](LICENSE)

## Acknowledgments

- This code is made in great part with the help of Onofre Martorell.
- Based on variational optical flow methods and deep unfolding techniques

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
