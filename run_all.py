"""
Run all 3 parts of CV Project 2 using the provided images (bg1.jpg, bg2.jpg).
Saves all results to the results/ folder.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
import os

os.makedirs('results', exist_ok=True)

# ============================================================
# Load images
# ============================================================
print("Loading images...")
bg1_full = cv2.imread('images/bg1.jpg')  # tablet on metallic surface
bg2_full = cv2.imread('images/bg2.jpg')  # building (H6 Bach Khoa)

# Resize for manageable processing
MAX_DIM = 800
for name, img in [('bg1', bg1_full), ('bg2', bg2_full)]:
    h, w = img.shape[:2]
    print(f"  {name}: {w}x{h}")

def resize_max(img, max_dim=MAX_DIM):
    h, w = img.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        return cv2.resize(img, (int(w * scale), int(h * scale)))
    return img

bg1 = resize_max(bg1_full)
bg2 = resize_max(bg2_full)

bg1_rgb = cv2.cvtColor(bg1, cv2.COLOR_BGR2RGB)
bg2_rgb = cv2.cvtColor(bg2, cv2.COLOR_BGR2RGB)

print(f"  Resized bg1: {bg1.shape[1]}x{bg1.shape[0]}")
print(f"  Resized bg2: {bg2.shape[1]}x{bg2.shape[0]}")


# ============================================================
# PART 1: Gradient Domain Editing (Poisson Blending)
# ============================================================
print("\n" + "="*60)
print("PART 1: Gradient Domain Editing")
print("="*60)

def get_neighbors(y, x):
    return [(y-1, x), (y+1, x), (y, x-1), (y, x+1)]

def naive_copy_paste(source, target, mask, offset):
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

def poisson_blend_channel(source_ch, target_ch, mask, offset):
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
                    pixel_to_idx[(y, x)] = len(omega_pixels)
                    omega_pixels.append((y, x))
    n = len(omega_pixels)
    if n == 0:
        return target_ch.copy()

    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for idx, (y, x) in enumerate(omega_pixels):
        ty, tx = y + oy, x + ox
        num_nb = 0
        for ny, nx in get_neighbors(y, x):
            nty, ntx = ny + oy, nx + ox
            if 0 <= nty < th and 0 <= ntx < tw:
                num_nb += 1
                if 0 <= ny < sh and 0 <= nx < sw:
                    b[idx] += source_ch[y, x] - source_ch[ny, nx]
                else:
                    b[idx] += source_ch[y, x]
                if (ny, nx) in pixel_to_idx:
                    A[idx, pixel_to_idx[(ny, nx)]] = -1
                else:
                    b[idx] += target_ch[nty, ntx]
        A[idx, idx] = num_nb

    A = csr_matrix(A)
    x_sol = spsolve(A, b)
    result = target_ch.copy()
    for idx, (y, xx) in enumerate(omega_pixels):
        result[y + oy, xx + ox] = np.clip(x_sol[idx], 0, 1)
    return result

def poisson_blend(source, target, mask, offset):
    result = np.zeros_like(target)
    for c in range(3):
        print(f"    Channel {c}...")
        result[:, :, c] = poisson_blend_channel(
            source[:, :, c], target[:, :, c], mask, offset)
    return result

def mixed_gradient_blend_channel(source_ch, target_ch, mask, offset):
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
                    pixel_to_idx[(y, x)] = len(omega_pixels)
                    omega_pixels.append((y, x))
    n = len(omega_pixels)
    if n == 0:
        return target_ch.copy()

    A = lil_matrix((n, n), dtype=np.float64)
    b = np.zeros(n, dtype=np.float64)
    for idx, (y, x) in enumerate(omega_pixels):
        ty, tx = y + oy, x + ox
        num_nb = 0
        for ny, nx in get_neighbors(y, x):
            nty, ntx = ny + oy, nx + ox
            if 0 <= nty < th and 0 <= ntx < tw:
                num_nb += 1
                gs = source_ch[y, x] - (source_ch[ny, nx] if 0 <= ny < sh and 0 <= nx < sw else 0)
                gt = target_ch[ty, tx] - target_ch[nty, ntx]
                b[idx] += gs if abs(gs) >= abs(gt) else gt
                if (ny, nx) in pixel_to_idx:
                    A[idx, pixel_to_idx[(ny, nx)]] = -1
                else:
                    b[idx] += target_ch[nty, ntx]
        A[idx, idx] = num_nb

    A = csr_matrix(A)
    x_sol = spsolve(A, b)
    result = target_ch.copy()
    for idx, (y, xx) in enumerate(omega_pixels):
        result[y + oy, xx + ox] = np.clip(x_sol[idx], 0, 1)
    return result

def mixed_gradient_blend(source, target, mask, offset):
    result = np.zeros_like(target)
    for c in range(3):
        print(f"    Channel {c}...")
        result[:, :, c] = mixed_gradient_blend_channel(
            source[:, :, c], target[:, :, c], mask, offset)
    return result

# --- Prepare source and target for Poisson blending ---
# Use bg2 (building) as target background
# Extract the tablet region from bg1 as source object to blend into building scene
target_img = bg2_rgb.astype(np.float64) / 255.0
th, tw = target_img.shape[:2]

# Crop the tablet region from bg1 as our source object
h1, w1 = bg1_rgb.shape[:2]
# Tablet is roughly in the center of bg1 - crop that region
tablet_y1, tablet_y2 = int(h1 * 0.2), int(h1 * 0.85)
tablet_x1, tablet_x2 = int(w1 * 0.15), int(w1 * 0.85)
source_crop = bg1_rgb[tablet_y1:tablet_y2, tablet_x1:tablet_x2]

# Resize source to fit nicely in the target
src_h = min(th // 3, source_crop.shape[0])
src_w = min(tw // 3, source_crop.shape[1])
scale = min(src_h / source_crop.shape[0], src_w / source_crop.shape[1])
source_img = cv2.resize(source_crop, (int(source_crop.shape[1] * scale),
                                       int(source_crop.shape[0] * scale)))
source_f = source_img.astype(np.float64) / 255.0
sh, sw = source_f.shape[:2]

# Create elliptical mask
mask = np.zeros((sh, sw), dtype=np.uint8)
cv2.ellipse(mask, (sw // 2, sh // 2), (sw // 2 - 10, sh // 2 - 10), 0, 0, 360, 255, -1)

# Place in center-right of the building
offset = (th // 3, tw // 2 - sw // 4)

print(f"  Source: {sw}x{sh}, Target: {tw}x{th}, Offset: {offset}")

# 1) Naive
print("  [1/4] Naive copy-paste...")
naive = naive_copy_paste(source_f, target_img, mask, offset)

# 2) Poisson
print("  [2/4] Poisson blending...")
poisson = poisson_blend(source_f, target_img, mask, offset)

# 3) Mixed gradient
print("  [3/4] Mixed gradient blending...")
mixed = mixed_gradient_blend(source_f, target_img, mask, offset)

# 4) OpenCV seamlessClone
print("  [4/4] OpenCV seamlessClone...")
oy, ox = offset
center = (ox + sw // 2, oy + sh // 2)
opencv_result = cv2.seamlessClone(
    cv2.cvtColor(source_img, cv2.COLOR_RGB2BGR),
    cv2.cvtColor((target_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    mask, center, cv2.NORMAL_CLONE
)
opencv_result = cv2.cvtColor(opencv_result, cv2.COLOR_BGR2RGB) / 255.0

# --- Plot Part 1 ---
fig, axes = plt.subplots(2, 3, figsize=(20, 13))
axes[0, 0].imshow(target_img); axes[0, 0].set_title('Target (Background)', fontsize=13); axes[0, 0].axis('off')
axes[0, 1].imshow(source_f); axes[0, 1].set_title('Source (Object)', fontsize=13); axes[0, 1].axis('off')
axes[0, 2].imshow(mask, cmap='gray'); axes[0, 2].set_title('Mask', fontsize=13); axes[0, 2].axis('off')
axes[1, 0].imshow(np.clip(naive, 0, 1)); axes[1, 0].set_title('Naive Copy-Paste', fontsize=13); axes[1, 0].axis('off')
axes[1, 1].imshow(np.clip(poisson, 0, 1)); axes[1, 1].set_title('Poisson Blending', fontsize=13); axes[1, 1].axis('off')
axes[1, 2].imshow(np.clip(opencv_result, 0, 1)); axes[1, 2].set_title('OpenCV seamlessClone', fontsize=13); axes[1, 2].axis('off')
plt.suptitle('Part 1: Gradient Domain Editing - Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/part1_gradient_domain_editing.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/part1_gradient_domain_editing.png")

# Additional: mixed gradient comparison
fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
axes2[0].imshow(np.clip(naive, 0, 1)); axes2[0].set_title('Naive (visible seams)', fontsize=13); axes2[0].axis('off')
axes2[1].imshow(np.clip(poisson, 0, 1)); axes2[1].set_title('Poisson Blending', fontsize=13); axes2[1].axis('off')
axes2[2].imshow(np.clip(mixed, 0, 1)); axes2[2].set_title('Mixed Gradient Blending', fontsize=13); axes2[2].axis('off')
plt.suptitle('Poisson vs Mixed Gradient', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/part1_mixed_gradient.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/part1_mixed_gradient.png")


# ============================================================
# PART 2: Geometric Transformations
# ============================================================
print("\n" + "="*60)
print("PART 2: Geometric Transformations")
print("="*60)

def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

def rotation_matrix(angle_deg, center=(0, 0)):
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    T2 = translation_matrix(cx, cy)
    return T2 @ R @ T1

def scaling_matrix(sx, sy, center=(0, 0)):
    cx, cy = center
    T1 = translation_matrix(-cx, -cy)
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    T2 = translation_matrix(cx, cy)
    return T2 @ S @ T1

# Use bg2 (building) for geometric transforms
img2 = bg2_rgb.copy()
h2, w2 = img2.shape[:2]
center2 = (w2 / 2, h2 / 2)
out_size2 = (w2 + 150, h2 + 150)

transforms = {}
transforms['Translation\n(tx=100, ty=60)'] = translation_matrix(100, 60)
transforms['Rotation\n(25° center)'] = rotation_matrix(25, center2)
transforms['Scaling\n(0.65x center)'] = scaling_matrix(0.65, 0.65, center2)
transforms['Rot+Scale\n(15°, 0.8x)'] = scaling_matrix(0.8, 0.8, center2) @ rotation_matrix(15, center2)

# Affine
src_pts = np.array([[50, 50], [w2-50, 50], [50, h2-50]], dtype=np.float32)
dst_pts = np.array([[80, 90], [w2-30, 70], [110, h2-20]], dtype=np.float32)
M_aff = cv2.getAffineTransform(src_pts, dst_pts)
transforms['Affine\n(shear+skew)'] = np.vstack([M_aff, [0, 0, 1]])

# Projective
src_h = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
dst_h = np.array([[60, 40], [w2-30, 70], [w2-90, h2-20], [40, h2-60]], dtype=np.float32)
H_proj, _ = cv2.findHomography(src_h, dst_h)
transforms['Projective\n(homography)'] = H_proj

# Plot all transforms
fig, axes = plt.subplots(2, 4, figsize=(24, 13))
axes[0, 0].imshow(img2)
axes[0, 0].set_title('Original', fontsize=13, fontweight='bold')
axes[0, 0].axis('off')

for idx, (name, H) in enumerate(transforms.items()):
    r, c = (idx + 1) // 4, (idx + 1) % 4
    result = cv2.warpPerspective(img2, H, out_size2, borderValue=(30, 30, 30))
    axes[r, c].imshow(result)
    axes[r, c].set_title(name, fontsize=11)
    axes[r, c].axis('off')

axes[1, 3].axis('off')
plt.suptitle('Part 2: Geometric Transformations on Building Image', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/part2_all_transformations.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/part2_all_transformations.png")

# --- Affine vs Projective detailed comparison ---
print("  Affine vs Projective comparison...")
src_corners = np.array([[0, 0], [w2, 0], [w2, h2], [0, h2]], dtype=np.float32)
dst_corners = np.array([
    [80, 60], [w2-80, 60], [w2+30, h2+30], [-30, h2+30]
], dtype=np.float32)

M_aff2 = cv2.getAffineTransform(src_corners[:3], dst_corners[:3])
H_aff2 = np.vstack([M_aff2, [0, 0, 1]])
H_proj2, _ = cv2.findHomography(src_corners, dst_corners)

out_size_cmp = (w2 + 120, h2 + 120)
res_aff = cv2.warpPerspective(img2, H_aff2, out_size_cmp, borderValue=(30, 30, 30))
res_proj = cv2.warpPerspective(img2, H_proj2, out_size_cmp, borderValue=(30, 30, 30))

fig, axes = plt.subplots(1, 3, figsize=(22, 7))
axes[0].imshow(img2); axes[0].set_title('Original', fontsize=14); axes[0].axis('off')
axes[1].imshow(res_aff); axes[1].set_title('Affine Transformation\n(parallel lines PRESERVED)', fontsize=13); axes[1].axis('off')
axes[2].imshow(res_proj); axes[2].set_title('Projective Transformation\n(parallel lines NOT preserved)', fontsize=13); axes[2].axis('off')
plt.suptitle('Part 2: Affine vs Projective Comparison', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/part2_affine_vs_projective.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/part2_affine_vs_projective.png")

# Print matrices
print(f"\n  Affine matrix (last row = [0, 0, 1]):")
print(f"  {np.round(H_aff2, 4)}")
print(f"\n  Projective matrix (last row != [0, 0, 1]):")
print(f"  {np.round(H_proj2, 4)}")

# --- Grid visualization ---
print("\n  Grid deformation visualization...")
def visualize_grid(H, title, ax, grid_size=10, extent=200):
    for i in range(0, extent + 1, grid_size):
        ax.plot([0, extent], [i, i], 'lightgray', lw=0.5)
        ax.plot([i, i], [0, extent], 'lightgray', lw=0.5)
    for i in range(0, extent + 1, grid_size):
        pts = np.array([[x, i, 1] for x in range(0, extent + 1, 2)])
        w = (H @ pts.T).T
        w = w[:, :2] / w[:, 2:3]
        ax.plot(w[:, 0], w[:, 1], 'b-', lw=0.8)
        pts = np.array([[i, y, 1] for y in range(0, extent + 1, 2)])
        w = (H @ pts.T).T
        w = w[:, :2] / w[:, 2:3]
        ax.plot(w[:, 0], w[:, 1], 'r-', lw=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xlim(-50, extent + 100)
    ax.set_ylim(-50, extent + 100)
    ax.invert_yaxis()

fig, axes = plt.subplots(1, 3, figsize=(20, 7))
visualize_grid(np.eye(3), 'Original Grid', axes[0])
visualize_grid(H_aff2, 'Affine Warp\n(parallelism preserved)', axes[1])
visualize_grid(H_proj2, 'Projective Warp\n(perspective distortion)', axes[2])
plt.suptitle('Grid Deformation: Affine vs Projective', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('results/part2_grid_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/part2_grid_comparison.png")


# ============================================================
# PART 3: Projective Billboard Pasting
# ============================================================
print("\n" + "="*60)
print("PART 3: Projective Billboard Pasting")
print("="*60)

def compute_homography_dlt(src_pts, dst_pts):
    """Manual DLT for educational verification."""
    n = len(src_pts)
    A = np.zeros((2 * n, 9))
    for i in range(n):
        x, y = src_pts[i]
        xp, yp = dst_pts[i]
        A[2*i]   = [-x, -y, -1,  0,  0,  0, xp*x, xp*y, xp]
        A[2*i+1] = [ 0,  0,  0, -x, -y, -1, yp*x, yp*y, yp]
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def paste_on_surface(scene, content, target_corners, blend_mode='replace'):
    ch, cw = content.shape[:2]
    sh, sw = scene.shape[:2]
    src_corners = np.array([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]], dtype=np.float32)
    target_corners = np.array(target_corners, dtype=np.float32)
    H = cv2.getPerspectiveTransform(src_corners, target_corners)
    warped = cv2.warpPerspective(content, H, (sw, sh))
    wmask = cv2.warpPerspective(np.ones((ch, cw), dtype=np.uint8) * 255, H, (sw, sh))

    result = scene.copy()
    if blend_mode == 'alpha':
        kern = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(wmask, kern, iterations=3)
        alpha = cv2.GaussianBlur(eroded, (21, 21), 10).astype(np.float32) / 255.0
        for c in range(3):
            result[:, :, c] = (alpha * warped[:, :, c] + (1 - alpha) * scene[:, :, c]).astype(np.uint8)
    else:
        m = wmask > 0
        for c in range(3):
            result[:, :, c][m] = warped[:, :, c][m]
    return result, warped, wmask

# --- Content image to paste (colorful poster) ---
def make_poster(h=300, w=450):
    poster = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        r = y / h
        poster[y, :] = [int(220 - 180*r), int(50 + 150*r), int(30 + 200*r)]
    cv2.putText(poster, 'BACH KHOA', (w//6, h//3), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    cv2.putText(poster, 'UNIVERSITY', (w//6 - 10, 2*h//3), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 0), 3)
    cv2.rectangle(poster, (10, 10), (w-10, h-10), (255, 255, 255), 3)
    return poster

content = make_poster()
content_rgb = cv2.cvtColor(content, cv2.COLOR_BGR2RGB)

# === Scene 1: Tablet screen (bg1) ===
print("  [1/2] Pasting onto tablet screen (bg1)...")
scene1 = bg1_rgb.copy()
h1, w1 = scene1.shape[:2]

# Tablet screen corners (manually identified from the image)
# The tablet is tilted, screen area is roughly:
tablet_corners = np.array([
    [int(w1 * 0.22), int(h1 * 0.22)],   # top-left of screen
    [int(w1 * 0.78), int(h1 * 0.15)],   # top-right of screen
    [int(w1 * 0.82), int(h1 * 0.78)],   # bottom-right of screen
    [int(w1 * 0.28), int(h1 * 0.88)],   # bottom-left of screen
], dtype=np.float32)

res_tablet, _, _ = paste_on_surface(scene1, content_rgb, tablet_corners, 'replace')
res_tablet_blend, _, _ = paste_on_surface(scene1, content_rgb, tablet_corners, 'alpha')

# === Scene 2: Building wall (bg2) ===
print("  [2/2] Pasting onto building wall (bg2)...")
scene2 = bg2_rgb.copy()
h2, w2 = scene2.shape[:2]

# Large wall area on the building - the blue flat wall section
wall_corners = np.array([
    [int(w2 * 0.05), int(h2 * 0.45)],   # top-left
    [int(w2 * 0.55), int(h2 * 0.10)],   # top-right (perspective)
    [int(w2 * 0.55), int(h2 * 0.55)],   # bottom-right
    [int(w2 * 0.05), int(h2 * 0.80)],   # bottom-left
], dtype=np.float32)

res_wall, _, _ = paste_on_surface(scene2, content_rgb, wall_corners, 'replace')
res_wall_blend, _, _ = paste_on_surface(scene2, content_rgb, wall_corners, 'alpha')

# --- Plot Part 3 ---
fig, axes = plt.subplots(2, 3, figsize=(22, 14))

# Show content
axes[0, 0].imshow(content_rgb)
axes[0, 0].set_title('Content to Paste', fontsize=13)
axes[0, 0].axis('off')

# Tablet scene with marked corners
scene1_marked = scene1.copy()
pts = tablet_corners.astype(np.int32)
for i in range(4):
    cv2.circle(scene1_marked, tuple(pts[i]), 8, (255, 0, 0), -1)
    cv2.line(scene1_marked, tuple(pts[i]), tuple(pts[(i+1)%4]), (255, 0, 0), 3)
axes[0, 1].imshow(scene1_marked)
axes[0, 1].set_title('Tablet - Target Region', fontsize=13)
axes[0, 1].axis('off')

# Building scene with marked corners
scene2_marked = scene2.copy()
pts2 = wall_corners.astype(np.int32)
for i in range(4):
    cv2.circle(scene2_marked, tuple(pts2[i]), 8, (255, 0, 0), -1)
    cv2.line(scene2_marked, tuple(pts2[i]), tuple(pts2[(i+1)%4]), (255, 0, 0), 3)
axes[0, 2].imshow(scene2_marked)
axes[0, 2].set_title('Building - Target Region', fontsize=13)
axes[0, 2].axis('off')

# Results
axes[1, 0].imshow(res_tablet)
axes[1, 0].set_title('Tablet: Direct Paste', fontsize=13)
axes[1, 0].axis('off')

axes[1, 1].imshow(res_tablet_blend)
axes[1, 1].set_title('Tablet: Alpha Blend', fontsize=13)
axes[1, 1].axis('off')

axes[1, 2].imshow(res_wall_blend)
axes[1, 2].set_title('Building Wall: Alpha Blend', fontsize=13)
axes[1, 2].axis('off')

plt.suptitle('Part 3: Projective Billboard Pasting', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('results/part3_billboard_pasting.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: results/part3_billboard_pasting.png")

# --- DLT Verification ---
ch, cw = content_rgb.shape[:2]
src_c = np.array([[0, 0], [cw-1, 0], [cw-1, ch-1], [0, ch-1]], dtype=np.float32)

H_cv = cv2.getPerspectiveTransform(src_c, tablet_corners)
H_dlt = compute_homography_dlt(src_c, tablet_corners)

print(f"\n  Homography (OpenCV):\n  {np.round(H_cv, 4)}")
print(f"\n  Homography (Manual DLT):\n  {np.round(H_dlt, 4)}")
print(f"\n  Max difference: {np.max(np.abs(H_cv - H_dlt)):.8f}")


# ============================================================
# DONE
# ============================================================
print("\n" + "="*60)
print("ALL DONE! Results saved in results/ folder:")
print("="*60)
for f in sorted(os.listdir('results')):
    fpath = os.path.join('results', f)
    size_kb = os.path.getsize(fpath) / 1024
    print(f"  {f} ({size_kb:.0f} KB)")
