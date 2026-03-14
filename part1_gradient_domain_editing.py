"""
Part 1: Gradient Domain Editing (Poisson Image Blending)
=========================================================
Implements seamless image cloning using Poisson equation solving.
The goal is to blend a source object into a target background naturally,
preserving gradients of the source while matching boundary conditions.

Theory:
    Given source image g, target image f*, and region Omega with boundary dOmega,
    we solve for f inside Omega:
        min_f  sum_{p in Omega} sum_{q in N(p)} (f_p - f_q - (g_p - g_q))^2
    subject to: f_p = f*_p for p on dOmega

    This leads to the Poisson equation:  nabla^2 f = nabla^2 g
    Discretized as a sparse linear system: Af = b
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import os

# ============================================================
# Utility Functions
# ============================================================

def load_and_resize(path, target_shape=None):
    """Load an image and optionally resize it."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if target_shape is not None:
        img = cv2.resize(img, (target_shape[1], target_shape[0]))
    return img.astype(np.float64) / 255.0


def create_elliptical_mask(h, w, center=None, axes=None):
    """Create an elliptical binary mask."""
    if center is None:
        center = (w // 2, h // 2)
    if axes is None:
        axes = (w // 3, h // 3)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def create_rectangular_mask(h, w, margin_ratio=0.15):
    """Create a rectangular binary mask with given margin."""
    mask = np.zeros((h, w), dtype=np.uint8)
    mx = int(w * margin_ratio)
    my = int(h * margin_ratio)
    mask[my:h-my, mx:w-mx] = 255
    return mask


# ============================================================
# Naive Copy-Paste (for comparison)
# ============================================================

def naive_copy_paste(source, target, mask, offset=(0, 0)):
    """
    Directly copy pixels from source to target using mask.
    This produces visible seams at the boundary.
    """
    result = target.copy()
    oy, ox = offset
    sh, sw = source.shape[:2]
    th, tw = target.shape[:2]

    for y in range(sh):
        for x in range(sw):
            ty, tx = y + oy, x + ox
            if 0 <= ty < th and 0 <= tx < tw and mask[y, x] > 0:
                result[ty, tx] = source[y, x]
    return result


# ============================================================
# Poisson Blending (Core Algorithm)
# ============================================================

def get_neighbors(y, x):
    """Return 4-connected neighbors of pixel (y, x)."""
    return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]


def poisson_blend_channel(source_ch, target_ch, mask, offset):
    """
    Solve Poisson equation for a single channel.

    For each pixel p inside Omega (mask region):
        |N(p)| * f_p - sum_{q in N(p) & Omega} f_q =
            sum_{q in N(p)} (g_p - g_q) + sum_{q in N(p) & dOmega} f*_q

    This creates a sparse linear system Ax = b where:
        A: Laplacian matrix (sparse, symmetric positive definite)
        x: unknown pixel values inside Omega
        b: divergence of source gradient + boundary values from target
    """
    oy, ox = offset
    sh, sw = source_ch.shape
    th, tw = target_ch.shape

    # Find all pixels inside the mask region
    omega_pixels = []
    pixel_to_idx = {}

    for y in range(sh):
        for x in range(sw):
            if mask[y, x] > 0:
                ty, tx = y + oy, x + ox
                if 1 <= ty < th - 1 and 1 <= tx < tw - 1:
                    idx = len(omega_pixels)
                    omega_pixels.append((y, x))
                    pixel_to_idx[(y, x)] = idx

    n = len(omega_pixels)
    if n == 0:
        return target_ch.copy()

    # Build sparse linear system
    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for idx, (y, x) in enumerate(omega_pixels):
        ty, tx = y + oy, x + ox
        num_neighbors = 0

        for ny, nx in get_neighbors(y, x):
            nty, ntx = ny + oy, nx + ox

            # Check if neighbor is within target bounds
            if 0 <= nty < th and 0 <= ntx < tw:
                num_neighbors += 1

                # Gradient guidance from source: v_pq = g_p - g_q
                b[idx] += source_ch[y, x] - source_ch[ny, nx] if (
                    0 <= ny < sh and 0 <= nx < sw) else source_ch[y, x]

                if (ny, nx) in pixel_to_idx:
                    # Neighbor is also in Omega -> add to matrix
                    j = pixel_to_idx[(ny, nx)]
                    A[idx, j] = -1
                else:
                    # Neighbor is on boundary -> use target value
                    b[idx] += target_ch[nty, ntx]

        A[idx, idx] = num_neighbors

    # Solve the sparse system
    A = csr_matrix(A)
    x = spsolve(A, b)

    # Write solution back into the result
    result = target_ch.copy()
    for idx, (y, xx) in enumerate(omega_pixels):
        ty, tx = y + oy, xx + ox
        result[ty, tx] = np.clip(x[idx], 0, 1)

    return result


def poisson_blend(source, target, mask, offset=(0, 0)):
    """
    Perform Poisson blending across all color channels.

    Args:
        source: Source image (H_s x W_s x 3), float [0, 1]
        target: Target/background image (H_t x W_t x 3), float [0, 1]
        mask: Binary mask (H_s x W_s), uint8 (0 or 255)
        offset: (y_offset, x_offset) position of source in target

    Returns:
        Blended image (H_t x W_t x 3), float [0, 1]
    """
    result = np.zeros_like(target)
    for c in range(3):
        print(f"  Solving Poisson equation for channel {c}...")
        result[:, :, c] = poisson_blend_channel(
            source[:, :, c], target[:, :, c], mask, offset
        )
    return result


# ============================================================
# Mixed Gradient Blending (Enhancement)
# ============================================================

def mixed_gradient_blend_channel(source_ch, target_ch, mask, offset):
    """
    Mixed gradient variant: at each pixel, use the gradient with
    larger magnitude (from either source or target).
    This preserves strong edges from both images.
    """
    oy, ox = offset
    sh, sw = source_ch.shape
    th, tw = target_ch.shape

    omega_pixels = []
    pixel_to_idx = {}

    for y in range(sh):
        for x in range(sw):
            if mask[y, x] > 0:
                ty, tx = y + oy, x + ox
                if 1 <= ty < th - 1 and 1 <= tx < tw - 1:
                    idx = len(omega_pixels)
                    omega_pixels.append((y, x))
                    pixel_to_idx[(y, x)] = idx

    n = len(omega_pixels)
    if n == 0:
        return target_ch.copy()

    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)

    for idx, (y, x) in enumerate(omega_pixels):
        ty, tx = y + oy, x + ox
        num_neighbors = 0

        for ny, nx in get_neighbors(y, x):
            nty, ntx = ny + oy, nx + ox

            if 0 <= nty < th and 0 <= ntx < tw:
                num_neighbors += 1

                # Compute gradients from both source and target
                grad_source = source_ch[y, x] - (
                    source_ch[ny, nx] if 0 <= ny < sh and 0 <= nx < sw else 0)
                grad_target = target_ch[ty, tx] - target_ch[nty, ntx]

                # Use the gradient with larger magnitude (mixed gradient)
                if abs(grad_source) >= abs(grad_target):
                    v_pq = grad_source
                else:
                    v_pq = grad_target

                b[idx] += v_pq

                if (ny, nx) in pixel_to_idx:
                    j = pixel_to_idx[(ny, nx)]
                    A[idx, j] = -1
                else:
                    b[idx] += target_ch[nty, ntx]

        A[idx, idx] = num_neighbors

    A = csr_matrix(A)
    x = spsolve(A, b)

    result = target_ch.copy()
    for idx, (y, xx) in enumerate(omega_pixels):
        ty, tx = y + oy, xx + ox
        result[ty, tx] = np.clip(x[idx], 0, 1)

    return result


def mixed_gradient_blend(source, target, mask, offset=(0, 0)):
    """Perform mixed gradient blending across all channels."""
    result = np.zeros_like(target)
    for c in range(3):
        print(f"  Solving mixed gradient for channel {c}...")
        result[:, :, c] = mixed_gradient_blend_channel(
            source[:, :, c], target[:, :, c], mask, offset
        )
    return result


# ============================================================
# Demo with Sample Images
# ============================================================

def generate_sample_images():
    """Generate sample source and target images for demonstration."""
    # Target: a sky/landscape gradient
    target = np.zeros((400, 600, 3), dtype=np.float64)
    for y in range(400):
        # Sky gradient (blue to light blue)
        ratio = y / 400.0
        if y < 200:
            target[y, :] = [0.3 + 0.4 * ratio, 0.5 + 0.3 * ratio, 0.9 - 0.2 * ratio]
        else:
            # Ground (green)
            target[y, :] = [0.2 + 0.1 * ratio, 0.6 - 0.2 * ratio, 0.1 + 0.1 * ratio]

    # Source: a bright circle (sun/object) with surrounding area
    source = np.zeros((150, 150, 3), dtype=np.float64)
    cy, cx = 75, 75
    for y in range(150):
        for x in range(150):
            dist = np.sqrt((y - cy)**2 + (x - cx)**2)
            if dist < 50:
                # Bright yellow-orange object
                intensity = 1.0 - (dist / 50) * 0.3
                source[y, x] = [intensity, intensity * 0.85, intensity * 0.2]
            else:
                # Dark surrounding
                source[y, x] = [0.1, 0.1, 0.15]

    # Mask: elliptical region
    mask = create_elliptical_mask(150, 150, (75, 75), (45, 45))

    return source, target, mask


def demo_with_files(source_path, target_path, mask_path=None, offset=(50, 200)):
    """Run the full demo with image files."""
    print("Loading images...")
    target = load_and_resize(target_path)
    source = load_and_resize(source_path)

    # Resize source if too large relative to target
    sh, sw = source.shape[:2]
    th, tw = target.shape[:2]
    if sh > th // 2 or sw > tw // 2:
        scale = min(th // 2 / sh, tw // 2 / sw)
        source = cv2.resize(source, (int(sw * scale), int(sh * scale)))

    sh, sw = source.shape[:2]

    if mask_path:
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (sw, sh))
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    else:
        mask = create_elliptical_mask(sh, sw)

    return run_comparison(source, target, mask, offset)


def run_comparison(source, target, mask, offset=(50, 200)):
    """Run naive vs Poisson vs mixed gradient comparison."""
    print("\n=== Gradient Domain Editing Demo ===\n")

    # 1) Naive copy-paste
    print("[1/3] Naive copy-paste...")
    naive = naive_copy_paste(source, target, mask, offset)

    # 2) Poisson blending
    print("[2/3] Poisson blending (seamless cloning)...")
    poisson = poisson_blend(source, target, mask, offset)

    # 3) Mixed gradient blending
    print("[3/3] Mixed gradient blending...")
    mixed = mixed_gradient_blend(source, target, mask, offset)

    # Also compare with OpenCV's seamlessClone if available
    print("\n[Bonus] OpenCV seamlessClone for reference...")
    try:
        src_u8 = (source * 255).astype(np.uint8)
        tgt_u8 = (target * 255).astype(np.uint8)
        oy, ox = offset
        center = (ox + source.shape[1] // 2, oy + source.shape[0] // 2)

        # OpenCV expects BGR
        opencv_result = cv2.seamlessClone(
            cv2.cvtColor(src_u8, cv2.COLOR_RGB2BGR),
            cv2.cvtColor(tgt_u8, cv2.COLOR_RGB2BGR),
            mask, center, cv2.NORMAL_CLONE
        )
        opencv_result = cv2.cvtColor(opencv_result, cv2.COLOR_BGR2RGB) / 255.0
        has_opencv = True
    except Exception as e:
        print(f"  OpenCV seamlessClone failed: {e}")
        opencv_result = None
        has_opencv = False

    # Visualization
    n_plots = 5 + (1 if has_opencv else 0)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(target)
    axes[0, 0].set_title("Target (Background)", fontsize=14)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(source)
    axes[0, 1].set_title("Source", fontsize=14)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(mask, cmap="gray")
    axes[0, 2].set_title("Mask", fontsize=14)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(np.clip(naive, 0, 1))
    axes[1, 0].set_title("Naive Copy-Paste", fontsize=14)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(np.clip(poisson, 0, 1))
    axes[1, 1].set_title("Poisson Blending (Ours)", fontsize=14)
    axes[1, 1].axis("off")

    if has_opencv:
        axes[1, 2].imshow(np.clip(opencv_result, 0, 1))
        axes[1, 2].set_title("OpenCV seamlessClone", fontsize=14)
    else:
        axes[1, 2].imshow(np.clip(mixed, 0, 1))
        axes[1, 2].set_title("Mixed Gradient Blending", fontsize=14)
    axes[1, 2].axis("off")

    plt.suptitle("Gradient Domain Editing: Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/part1_gradient_domain_editing.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("\nResults saved to results/part1_gradient_domain_editing.png")

    return naive, poisson, mixed


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    # Check if user-provided images exist
    source_path = "images/source.jpg"
    target_path = "images/background.jpg"

    if os.path.exists(source_path) and os.path.exists(target_path):
        mask_path = "images/mask.png" if os.path.exists("images/mask.png") else None
        demo_with_files(source_path, target_path, mask_path)
    else:
        print("No custom images found. Using generated sample images.")
        print("To use your own images, place them in the images/ folder:")
        print("  - images/source.jpg   (object to paste)")
        print("  - images/background.jpg  (target scene)")
        print("  - images/mask.png  (optional, binary mask)")
        print()

        source, target, mask = generate_sample_images()
        run_comparison(source, target, mask, offset=(50, 200))
