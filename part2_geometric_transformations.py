"""
Part 2: Geometric Transformations
===================================
Demonstrates fundamental geometric transformations in Computer Vision:
    - Translation
    - Rotation
    - Scaling
    - Affine Transformation
    - Projective (Homography) Transformation

    Theory:
    A 2D point (x, y) is represented in homogeneous coordinates as (x, y, 1).
    Transformations are applied via matrix multiplication:
        [x']   [m11  m12  m13] [x]
        [y'] = [m21  m22  m23] [y]
        [w']   [m31  m32  m33] [1]
    Final coordinates: (x'/w', y'/w')

    - Translation (2 DOF):     [[1, 0, tx], [0, 1, ty], [0, 0, 1]]
    - Rotation (1 DOF):        [[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]]
    - Scaling (2 DOF):         [[sx, 0, 0], [0, sy, 0], [0, 0, 1]]
    - Affine (6 DOF):          [[a11, a12, a13], [a21, a22, a23], [0, 0, 1]]
        -> Preserves parallel lines, ratios of distances along lines
    - Projective (8 DOF):      [[h11, h12, h13], [h21, h22, h23], [h31, h32, 1]]
        -> Preserves straight lines but NOT parallelism
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# ============================================================
# Transformation Matrix Builders
# ============================================================

def translation_matrix(tx, ty):
    """
    Translation matrix:
        [1  0  tx]
        [0  1  ty]
        [0  0   1]
    Shifts every point by (tx, ty).
    """
    return np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0,  1]
    ], dtype=np.float64)


def rotation_matrix(angle_deg, center=(0, 0)):
    """
    Rotation matrix around a given center:
        T(center) * R(theta) * T(-center)
    where R = [[cos, -sin], [sin, cos]]
    """
    theta = np.radians(angle_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cx, cy = center

    # Translate to origin, rotate, translate back
    T1 = translation_matrix(-cx, -cy)
    R = np.array([
        [cos_t, -sin_t, 0],
        [sin_t,  cos_t, 0],
        [0,      0,     1]
    ], dtype=np.float64)
    T2 = translation_matrix(cx, cy)

    return T2 @ R @ T1


def scaling_matrix(sx, sy, center=(0, 0)):
    """
    Scaling matrix around a given center:
        T(center) * S(sx, sy) * T(-center)
    """
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    S = np.array([
        [sx, 0, 0],
        [0, sy, 0],
        [0,  0, 1]
    ], dtype=np.float64)
    T2 = translation_matrix(cx, cy)

    return T2 @ S @ T1


def affine_from_points(src_pts, dst_pts):
    """
    Compute affine transformation matrix from 3 point correspondences.

    Affine transform has 6 DOF, so 3 point pairs (6 equations) suffice.
    The last row is always [0, 0, 1] -> preserves parallelism.

    Args:
        src_pts: 3x2 array of source points
        dst_pts: 3x2 array of destination points

    Returns:
        3x3 affine transformation matrix
    """
    M = cv2.getAffineTransform(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32)
    )
    # Convert 2x3 to 3x3
    H = np.vstack([M, [0, 0, 1]])
    return H


def homography_from_points(src_pts, dst_pts):
    """
    Compute projective (homography) matrix from 4 point correspondences.

    Homography has 8 DOF, so 4 point pairs (8 equations) are needed.
    Unlike affine, the last row is NOT [0, 0, 1].
    This allows mapping any quadrilateral to any other quadrilateral.

    Uses DLT (Direct Linear Transform) algorithm internally.

    Args:
        src_pts: 4x2 array of source points
        dst_pts: 4x2 array of destination points

    Returns:
        3x3 homography matrix
    """
    H, _ = cv2.findHomography(
        src_pts.astype(np.float32),
        dst_pts.astype(np.float32)
    )
    return H


# ============================================================
# Manual Warping Functions (for educational purposes)
# ============================================================

def apply_transform_manual(image, H, output_shape):
    """
    Apply a 3x3 transformation matrix using inverse warping.

    For each pixel (x', y') in the output, find the corresponding
    source pixel (x, y) using H_inv, then sample from the input image.

    This avoids holes that would occur with forward warping.
    """
    h_out, w_out = output_shape[:2]
    h_in, w_in = image.shape[:2]
    channels = image.shape[2] if len(image.shape) == 3 else 1

    H_inv = np.linalg.inv(H)

    if channels > 1:
        result = np.zeros((h_out, w_out, channels), dtype=image.dtype)
    else:
        result = np.zeros((h_out, w_out), dtype=image.dtype)

    for y_out in range(h_out):
        for x_out in range(w_out):
            # Inverse transform: find source coordinates
            p = H_inv @ np.array([x_out, y_out, 1.0])
            x_in = p[0] / p[2]
            y_in = p[1] / p[2]

            # Bilinear interpolation
            if 0 <= x_in < w_in - 1 and 0 <= y_in < h_in - 1:
                x0, y0 = int(x_in), int(y_in)
                dx, dy = x_in - x0, y_in - y0

                val = ((1-dy) * (1-dx) * image[y0, x0] +
                       (1-dy) * dx * image[y0, x0+1] +
                       dy * (1-dx) * image[y0+1, x0] +
                       dy * dx * image[y0+1, x0+1])
                result[y_out, x_out] = val

    return result


# ============================================================
# Visualization Functions
# ============================================================

def visualize_grid_transform(H, title, ax, grid_size=10, extent=200):
    """Visualize how a transformation warps a regular grid."""
    # Draw original grid in light gray
    for i in range(0, extent + 1, grid_size):
        ax.plot([0, extent], [i, i], 'lightgray', linewidth=0.5)
        ax.plot([i, i], [0, extent], 'lightgray', linewidth=0.5)

    # Transform and draw warped grid in blue
    for i in range(0, extent + 1, grid_size):
        # Horizontal line
        pts = np.array([[x, i, 1] for x in range(0, extent + 1, 2)])
        warped = (H @ pts.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        ax.plot(warped[:, 0], warped[:, 1], 'b-', linewidth=0.8)

        # Vertical line
        pts = np.array([[i, y, 1] for y in range(0, extent + 1, 2)])
        warped = (H @ pts.T).T
        warped = warped[:, :2] / warped[:, 2:3]
        ax.plot(warped[:, 0], warped[:, 1], 'r-', linewidth=0.8)

    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.set_xlim(-50, extent + 100)
    ax.set_ylim(-50, extent + 100)
    ax.invert_yaxis()


def generate_checkerboard(size=400, squares=8):
    """Generate a checkerboard pattern for clear visualization."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    sq = size // squares
    colors = [
        (200, 50, 50),   # Red
        (50, 200, 50),   # Green
        (50, 50, 200),   # Blue
        (200, 200, 50),  # Yellow
    ]

    for i in range(squares):
        for j in range(squares):
            color_idx = (i + j) % 2
            base_color = colors[color_idx * 2 + (i // (squares // 2))]
            cv2.rectangle(img,
                          (j * sq, i * sq),
                          ((j + 1) * sq, (i + 1) * sq),
                          base_color, -1)

    # Add some text/markers for orientation
    cv2.putText(img, "TL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "TR", (size - 50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "BL", (10, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(img, "BR", (size - 50, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Add a circle for reference
    cv2.circle(img, (size // 2, size // 2), size // 6, (255, 255, 255), 3)

    return img


def load_or_generate_image(path="images/input_transform.jpg", size=400):
    """Load user image or generate a test pattern."""
    if os.path.exists(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (size, size))
        return img
    else:
        return generate_checkerboard(size)


# ============================================================
# Main Demo
# ============================================================

def demo_all_transformations():
    """Demonstrate all geometric transformations on the same input image."""
    print("\n=== Geometric Transformations Demo ===\n")

    img = load_or_generate_image()
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # Define transformations
    transforms = {}

    # 1. Translation
    tx, ty = 80, 50
    transforms["Translation\n(tx=80, ty=50)"] = translation_matrix(tx, ty)

    # 2. Rotation (30 degrees around center)
    transforms["Rotation\n(30° around center)"] = rotation_matrix(30, center)

    # 3. Scaling (0.7x around center)
    transforms["Scaling\n(0.7x around center)"] = scaling_matrix(0.7, 0.7, center)

    # 4. Combined: Rotation + Scaling
    R = rotation_matrix(20, center)
    S = scaling_matrix(0.8, 0.8, center)
    transforms["Rotation + Scaling\n(20°, 0.8x)"] = S @ R

    # 5. Affine transformation
    src_pts = np.array([[50, 50], [w-50, 50], [50, h-50]], dtype=np.float32)
    dst_pts = np.array([[70, 80], [w-30, 60], [100, h-30]], dtype=np.float32)
    M_affine = cv2.getAffineTransform(src_pts, dst_pts)
    H_affine = np.vstack([M_affine, [0, 0, 1]])
    transforms["Affine\n(shear + skew)"] = H_affine

    # 6. Projective (Homography) transformation
    src_pts_h = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst_pts_h = np.array([[50, 30], [w-20, 60], [w-80, h-20], [30, h-50]], dtype=np.float32)
    H_proj, _ = cv2.findHomography(src_pts_h, dst_pts_h)
    transforms["Projective\n(homography)"] = H_proj

    # Apply all transformations and display
    fig, axes = plt.subplots(2, 4, figsize=(22, 12))
    output_size = (w + 100, h + 100)  # Slightly larger output to see the full transform

    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("Original", fontsize=13, fontweight="bold")
    axes[0, 0].axis("off")

    for idx, (name, H) in enumerate(transforms.items()):
        row = (idx + 1) // 4
        col = (idx + 1) % 4

        # Use OpenCV warpPerspective (fast)
        result = cv2.warpPerspective(img, H, output_size,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(30, 30, 30))

        axes[row, col].imshow(result)
        axes[row, col].set_title(name, fontsize=11)
        axes[row, col].axis("off")

    # Hide last subplot if unused
    if len(transforms) + 1 < 8:
        axes[1, 3].axis("off")

    plt.suptitle("Geometric Transformations Comparison", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/part2_all_transformations.png", dpi=150, bbox_inches="tight")
    plt.show()

    return transforms


def demo_affine_vs_projective():
    """
    Detailed comparison between Affine and Projective transformations.
    Key difference: Affine preserves parallel lines, Projective does not.
    """
    print("\n=== Affine vs Projective Comparison ===\n")

    img = load_or_generate_image()
    h, w = img.shape[:2]

    # Same 4 corners, different mappings
    src_corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    # Projective: converging lines (perspective effect)
    dst_proj = np.array([
        [80, 60],       # top-left moves right and down
        [w-80, 60],     # top-right moves left and down
        [w+20, h+20],   # bottom-right moves further out
        [-20, h+20]     # bottom-left moves further out
    ], dtype=np.float32)

    # Affine: use first 3 points only (parallel lines preserved)
    M_affine = cv2.getAffineTransform(src_corners[:3], dst_proj[:3])
    H_affine = np.vstack([M_affine, [0, 0, 1]])

    # Projective: use all 4 points
    H_proj, _ = cv2.findHomography(src_corners, dst_proj)

    out_size = (w + 100, h + 100)

    result_affine = cv2.warpPerspective(img, H_affine, out_size,
                                         borderValue=(30, 30, 30))
    result_proj = cv2.warpPerspective(img, H_proj, out_size,
                                       borderValue=(30, 30, 30))

    # Visualization with grid overlay
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    axes[0].imshow(img)
    axes[0].set_title("Original Image", fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(result_affine)
    axes[1].set_title("Affine Transformation\n(parallel lines preserved)", fontsize=12)
    axes[1].axis("off")

    axes[2].imshow(result_proj)
    axes[2].set_title("Projective Transformation\n(parallel lines NOT preserved)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle("Affine vs Projective: Key Differences",
                 fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/part2_affine_vs_projective.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Grid visualization
    print("\nGrid warp visualization...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))

    # Identity grid
    visualize_grid_transform(np.eye(3), "Original Grid", axes2[0])
    visualize_grid_transform(H_affine, "Affine Warp\n(parallelism preserved)", axes2[1])
    visualize_grid_transform(H_proj, "Projective Warp\n(perspective distortion)", axes2[2])

    plt.suptitle("Grid Deformation: Affine vs Projective", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/part2_grid_comparison.png", dpi=150, bbox_inches="tight")
    plt.show()

    print("\nKey observations:")
    print("  - Affine: parallel lines remain parallel after transformation")
    print("  - Projective: parallel lines can converge (vanishing point)")
    print("  - Affine is a special case of projective (last row = [0, 0, 1])")
    print("  - Affine needs 3 point pairs (6 DOF), Projective needs 4 (8 DOF)")

    # Print matrices for report
    print("\nAffine matrix:")
    print(np.round(H_affine, 4))
    print("\nProjective matrix:")
    print(np.round(H_proj, 4))
    print(f"\nNote: Affine last row = {H_affine[2, :]}")
    print(f"      Projective last row = {np.round(H_proj[2, :], 6)}")

    return H_affine, H_proj


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_all_transformations()
    demo_affine_vs_projective()
    print("\nAll results saved to results/ folder.")
