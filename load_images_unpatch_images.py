import tensorflow as tf
import numpy as np
from PIL import Image

# ============================
# IMAGE PATCHING FUNCTIONS
# ============================

IMAGE_PATH = "/kaggle/input/flickr30k-dataset/flickr30k/Images/1000092795.jpg"
IMG_SIZE = (128, 128)
PATCH_SIZE = 4


def load_image_as_patches(path, img_size=IMG_SIZE, patch_size=PATCH_SIZE):
    img = Image.open(path).resize(img_size)
    img = np.array(img, dtype=np.float32) / 255.0

    H, W, C = img.shape
    assert H % patch_size == 0 and W % patch_size == 0

    patches_h = H // patch_size
    patches_w = W // patch_size
    num_patches = patches_h * patches_w
    patch_dim = patch_size * patch_size * C

    patches = np.zeros((num_patches, patch_dim), dtype=np.float32)

    idx = 0
    for i in range(0, H, patch_size):
        for j in range(0, W, patch_size):
            patch = img[i:i+patch_size, j:j+patch_size, :]
            patches[idx] = patch.flatten()
            idx += 1

    return patches, (patches_h, patches_w), img.shape


def unpatchify(patches, grid_shape, patch_size=PATCH_SIZE):
    patches_h, patches_w = grid_shape
    H = patches_h * patch_size
    W = patches_w * patch_size
    C = 3

    img = np.zeros((H, W, C), dtype=np.float32)

    idx = 0
    for i in range(patches_h):
        for j in range(patches_w):
            patch = patches[idx].reshape(patch_size, patch_size, C)
            img[i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size] = patch
            idx += 1

    return (img * 255).clip(0, 255).astype(np.uint8)

from PIL import Image
import numpy as np

# Create a test image (128x128 RGB)
test_img = np.random.rand(128, 128, 3).astype(np.float32)
Image.fromarray((test_img*255).astype(np.uint8)).save("test_img.jpg")

# Load patches
patches, grid_shape, orig_shape = load_image_as_patches("test_img.jpg")
print("Patches shape:", patches.shape)  # (1024, 48) for PATCH_SIZE=4

# Reconstruct image
reconstructed_img = unpatchify(patches, grid_shape)
print("Reconstructed image shape:", reconstructed_img.shape)  # (128,128,3)

# Save reconstructed image
Image.fromarray(reconstructed_img).save("reconstructed_test_img.jpg")
