"""
Part 3: Projective Billboard Pasting (Extended Experiment)
============================================================
Paste a rectangular image (e.g., a portrait, advertisement) onto a
planar surface in a scene (e.g., billboard, wall, screen) using
projective (homography) transformation.

Pipeline:
    1. Load the scene image and the content image to paste.
    2. Define 4 corner points on the target planar surface.
    3. Compute homography from content corners to scene surface corners.
    4. Warp the content image using the homography.
    5. Blend into the scene using masking.

Theory:
    Given 4 correspondences between the content image corners and the
    target quadrilateral in the scene, we compute homography H such that:
        p_scene = H * p_content
    where points are in homogeneous coordinates.

    The homography H is computed using DLT (Direct Linear Transform):
        For each pair (x, x'), we get 2 equations:
            -x^T h1 + (x * x'^T) h3 = 0
            -x^T h2 + (y * x'^T) h3 = 0
        Stacked into Ah = 0, solved via SVD.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


# ============================================================
# Homography Computation (Manual DLT for educational purposes)
# ============================================================

def compute_homography_dlt(src_pts, dst_pts):
    """
    Compute homography using Direct Linear Transform (DLT).

    Given n >= 4 point correspondences, build the 2n x 9 matrix A
    and find h (flattened H) as the null space of A (via SVD).

    Each correspondence (x, y) -> (x', y') gives 2 rows:
        [-x, -y, -1,  0,  0,  0,  x'x, x'y, x']
        [ 0,  0,  0, -x, -y, -1,  y'x, y'y, y']

    Args:
        src_pts: Nx2 source points
        dst_pts: Nx2 destination points

    Returns:
        3x3 homography matrix
    """
    assert len(src_pts) >= 4, "Need at least 4 point correspondences"

    n = len(src_pts)
    A = np.zeros((2 * n, 9))

    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]

        A[2*i] = [-x, -y, -1, 0, 0, 0, xp*x, xp*y, xp]
        A[2*i+1] = [0, 0, 0, -x, -y, -1, yp*x, yp*y, yp]

    # SVD: A = U S V^T, solution is last row of V^T
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    # Normalize so H[2,2] = 1
    H = H / H[2, 2]
    return H


# ============================================================
# Billboard Pasting Functions
# ============================================================

def paste_on_surface(scene, content, target_corners, blend_mode="replace"):
    """
    Paste content image onto a planar surface in the scene.

    Args:
        scene: Scene/background image (H x W x 3)
        content: Content image to paste (h x w x 3)
        target_corners: 4 corners of target surface in scene,
                        ordered as [top-left, top-right, bottom-right, bottom-left]
        blend_mode: "replace" for direct paste, "alpha" for smooth blending

    Returns:
        Scene with content pasted on the surface
    """
    ch, cw = content.shape[:2]
    sh, sw = scene.shape[:2]

    # Source corners (content image corners)
    src_corners = np.array([
        [0, 0],
        [cw - 1, 0],
        [cw - 1, ch - 1],
        [0, ch - 1]
    ], dtype=np.float32)

    target_corners = np.array(target_corners, dtype=np.float32)

    # Compute homography
    H = cv2.getPerspectiveTransform(src_corners, target_corners)

    # Warp the content image
    warped_content = cv2.warpPerspective(content, H, (sw, sh))

    # Create a mask for the warped region
    mask = np.ones((ch, cw), dtype=np.uint8) * 255
    warped_mask = cv2.warpPerspective(mask, H, (sw, sh))

    # Blend into scene
    result = scene.copy()

    if blend_mode == "alpha":
        # Feathered blending at edges
        kernel = np.ones((5, 5), np.uint8)
        eroded_mask = cv2.erode(warped_mask, kernel, iterations=3)
        blurred_mask = cv2.GaussianBlur(eroded_mask, (21, 21), 10)
        alpha = blurred_mask.astype(np.float32) / 255.0

        for c in range(3):
            result[:, :, c] = (alpha * warped_content[:, :, c] +
                               (1 - alpha) * scene[:, :, c]).astype(np.uint8)
    else:
        # Direct replacement
        mask_bool = warped_mask > 0
        for c in range(3):
            result[:, :, c][mask_bool] = warped_content[:, :, c][mask_bool]

    return result, warped_content, warped_mask


def interactive_point_selector(scene_path):
    """
    Allow user to click 4 points on the scene image.
    Returns the 4 corner coordinates.
    """
    scene = cv2.imread(scene_path)
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append([x, y])
            cv2.circle(scene, (x, y), 5, (0, 255, 0), -1)
            if len(points) > 1:
                cv2.line(scene, tuple(points[-2]), tuple(points[-1]), (0, 255, 0), 2)
            if len(points) == 4:
                cv2.line(scene, tuple(points[3]), tuple(points[0]), (0, 255, 0), 2)
            cv2.imshow("Select 4 corners (TL, TR, BR, BL)", scene)

    cv2.imshow("Select 4 corners (TL, TR, BR, BL)", scene)
    cv2.setMouseCallback("Select 4 corners (TL, TR, BR, BL)", click_event)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return np.array(points, dtype=np.float32)


# ============================================================
# Demo Functions
# ============================================================

def generate_scene_image(size=(600, 800)):
    """Generate a synthetic scene with a visible planar surface (billboard)."""
    h, w = size
    scene = np.zeros((h, w, 3), dtype=np.uint8)

    # Sky gradient
    for y in range(h // 2):
        ratio = y / (h // 2)
        scene[y, :] = [int(180 - 80 * ratio), int(200 - 60 * ratio), int(240 - 40 * ratio)]

    # Ground
    for y in range(h // 2, h):
        ratio = (y - h // 2) / (h // 2)
        scene[y, :] = [int(80 + 30 * ratio), int(120 + 20 * ratio), int(80 + 30 * ratio)]

    # Building/wall
    cv2.rectangle(scene, (100, 100), (700, 500), (160, 150, 140), -1)
    cv2.rectangle(scene, (100, 100), (700, 500), (100, 90, 80), 3)

    # Windows
    for wx in [150, 300, 500, 600]:
        for wy in [150, 280, 380]:
            cv2.rectangle(scene, (wx, wy), (wx + 60, wy + 50), (200, 220, 240), -1)
            cv2.rectangle(scene, (wx, wy), (wx + 60, wy + 50), (80, 80, 80), 1)

    # Billboard frame on the building
    cv2.rectangle(scene, (195, 125), (555, 345), (60, 60, 60), 5)

    return scene


def generate_content_image(size=(300, 500)):
    """Generate a sample content image (advertisement/poster)."""
    h, w = size
    content = np.zeros((h, w, 3), dtype=np.uint8)

    # Gradient background
    for y in range(h):
        ratio = y / h
        content[y, :] = [int(30 + 200 * ratio), int(50 + 100 * (1-ratio)), int(200 - 150 * ratio)]

    # Add text
    cv2.putText(content, "HELLO", (w // 4 - 30, h // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    cv2.putText(content, "WORLD!", (w // 4 - 20, 2 * h // 3),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 0), 3)

    # Add a star
    star_center = (w // 2, h // 5)
    cv2.drawMarker(content, star_center, (255, 255, 100),
                   cv2.MARKER_STAR, 40, 3)

    return content


def demo_billboard_pasting():
    """Full demo of projective billboard pasting."""
    print("\n=== Projective Billboard Pasting Demo ===\n")

    # Check for user images
    scene_path = "images/scene.jpg"
    content_path = "images/content.jpg"

    if os.path.exists(scene_path) and os.path.exists(content_path):
        scene = cv2.imread(scene_path)
        scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        content = cv2.imread(content_path)
        content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)
        print("Loaded custom images.")
        # You would need to define target_corners for your specific scene
        # or use the interactive selector
        h, w = scene.shape[:2]
        target_corners = np.array([
            [w * 0.25, h * 0.2],
            [w * 0.75, h * 0.2],
            [w * 0.75, h * 0.6],
            [w * 0.25, h * 0.6]
        ], dtype=np.float32)
    else:
        print("Using generated sample images.")
        print("To use your own, place images/scene.jpg and images/content.jpg\n")
        scene = generate_scene_image()
        scene = cv2.cvtColor(scene, cv2.COLOR_BGR2RGB)
        content = generate_content_image()
        content = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)

        # Billboard corners on the generated scene (slightly perspective)
        target_corners = np.array([
            [200, 130],   # top-left
            [550, 130],   # top-right
            [550, 340],   # bottom-right
            [200, 340]    # bottom-left
        ], dtype=np.float32)

    # === Demo 1: Frontal paste (rectangle) ===
    print("[1/3] Frontal billboard paste...")
    result_frontal, warped1, mask1 = paste_on_surface(
        scene, content, target_corners, blend_mode="replace"
    )

    # === Demo 2: Perspective paste (angled surface) ===
    print("[2/3] Perspective billboard paste...")
    perspective_corners = target_corners.copy()
    # Add perspective effect: top edge narrower
    perspective_corners[0, 0] += 40
    perspective_corners[1, 0] -= 40
    perspective_corners[0, 1] += 15
    perspective_corners[1, 1] += 15

    result_perspective, warped2, mask2 = paste_on_surface(
        scene, content, perspective_corners, blend_mode="replace"
    )

    # === Demo 3: With alpha blending ===
    print("[3/3] Perspective paste with alpha blending...")
    result_blended, warped3, mask3 = paste_on_surface(
        scene, content, perspective_corners, blend_mode="alpha"
    )

    # === Visualization ===
    fig, axes = plt.subplots(2, 3, figsize=(20, 13))

    axes[0, 0].imshow(scene)
    axes[0, 0].set_title("Original Scene", fontsize=13)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(content)
    axes[0, 1].set_title("Content to Paste", fontsize=13)
    axes[0, 1].axis("off")

    # Show target corners on scene
    scene_with_corners = scene.copy()
    pts = target_corners.astype(np.int32)
    for i in range(4):
        cv2.circle(scene_with_corners, tuple(pts[i]), 6, (255, 0, 0), -1)
        cv2.line(scene_with_corners, tuple(pts[i]), tuple(pts[(i+1)%4]), (255, 0, 0), 2)
    axes[0, 2].imshow(scene_with_corners)
    axes[0, 2].set_title("Target Region (frontal)", fontsize=13)
    axes[0, 2].axis("off")

    axes[1, 0].imshow(result_frontal)
    axes[1, 0].set_title("Frontal Paste", fontsize=13)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(result_perspective)
    axes[1, 1].set_title("Perspective Paste", fontsize=13)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(result_blended)
    axes[1, 2].set_title("Perspective + Alpha Blend", fontsize=13)
    axes[1, 2].axis("off")

    plt.suptitle("Projective Billboard Pasting", fontsize=16, fontweight="bold")
    plt.tight_layout()

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/part3_billboard_pasting.png", dpi=150, bbox_inches="tight")
    plt.show()

    # === DLT Homography verification ===
    print("\n--- Homography Matrix Analysis ---")
    ch, cw = content.shape[:2]
    src_corners = np.array([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]], dtype=np.float32)

    H_opencv = cv2.getPerspectiveTransform(src_corners, perspective_corners)
    H_dlt = compute_homography_dlt(src_corners, perspective_corners)

    print("\nHomography (OpenCV):")
    print(np.round(H_opencv, 4))
    print("\nHomography (Manual DLT):")
    print(np.round(H_dlt, 4))
    print("\nDifference (should be ~0):")
    print(np.round(np.abs(H_opencv - H_dlt), 6))

    print("\nResults saved to results/part3_billboard_pasting.png")

    return result_frontal, result_perspective, result_blended


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    demo_billboard_pasting()
