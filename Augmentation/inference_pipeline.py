import argparse
import numpy as np
import torch
from monai.inferers import SlidingWindowSplitter, SlidingWindowInferer
from monai.networks.nets import SwinUNETR
from monai.networks.nets import DiNTS
from monai.networks.nets import TopologyInstance


def sliding_window_pipeline(input_volume, patch_size, overlap):
    splitter = SlidingWindowSplitter(
        patch_size=patch_size,
        overlap=overlap,
        pad_mode='constant',
        pad_value=0.0
    )

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=8,
        overlap=overlap,
        mode='gaussian',
        device='cpu'
    )

    # Split into patches
    patches = list(splitter(input_volume))
    print(f'Patch size: {patch_size}, overlap: {overlap}, num patches: {len(patches)}')
    for patch in patches:
        print(f'Real patch size: {patch[0].shape}, index: {patch[1]}')

    # Reconstruct using sliding window
    # inputs = torch.stack([p for p, _ in patches])
    # print(f'Input shape: {inputs.shape}')
    reconstructed = inferer(input_volume, lambda x: x)
    print(f'Reconstructed shape: {reconstructed.shape}')
    print(f'Reconstructed dtype: {reconstructed.dtype}')
    print(f'Reconstructed device: {reconstructed.device}')

    return reconstructed


def test_sliding_window_reconstruction(args):
    # Generate test data
    original = np.random.rand(*args.test_size).astype(np.float32)
    input_tensor = torch.rand((3, 1, 128, 128, 128))

    # Test parameters
    test_params = [
        ((64, 64, 64), 0.25),
        ((32, 32, 32), 0.5)
    ]

    for patch_size, overlap in test_params:
        reconstructed = sliding_window_pipeline(input_tensor, patch_size, overlap)

        # Calculate MSE
        mse = torch.mean((input_tensor - reconstructed) ** 2).item()
        print(f'Patch size {patch_size}, overlap {overlap}: MSE = {mse:.6f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, nargs=3, default=[64, 64, 64],
                        help='Sliding window size in 3D (e.g. 64 64 64)')
    parser.add_argument('--overlap', type=float, default=0.25,
                        help='Overlap ratio between windows (0.0-1.0)')
    parser.add_argument('--test_size', type=int, nargs=3, default=[128, 128, 128],
                        help='Test volume size in 3D (e.g. 128 128 128)')
    args = parser.parse_args()
    test_sliding_window_reconstruction(args)
