"""
Interactive Billboard Pasting Tool
====================================
Click 4 points on the scene image to define the target surface,
then the content image (bg3) is warped and pasted onto that surface.

Usage:
    python interactive_paste.py

Instructions:
    1. A window shows the scene image (bg1 or bg2).
    2. Click 4 corners in order: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    3. After 4 clicks, the content is warped and pasted. Press any key to continue.
    4. Press 'r' to reset points and try again.
    5. Press 's' to save the result.
    6. Press 'q' or ESC to quit.
"""

import numpy as np
import cv2
import os
import sys

MAX_DIM = 900  # Max display dimension


def resize_for_display(img, max_dim=MAX_DIM):
    """Resize image to fit screen while keeping aspect ratio."""
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / max(h, w)
    resized = cv2.resize(img, (int(w * scale), int(h * scale)))
    return resized, scale


class InteractivePaster:
    def __init__(self, scene_path, content_path):
        self.scene_orig = cv2.imread(scene_path)
        self.content_orig = cv2.imread(content_path)
        if self.scene_orig is None:
            raise FileNotFoundError(f"Cannot load scene: {scene_path}")
        if self.content_orig is None:
            raise FileNotFoundError(f"Cannot load content: {content_path}")

        self.scene_display, self.scale = resize_for_display(self.scene_orig)
        self.points = []
        self.result = None
        self.scene_name = os.path.basename(scene_path).split('.')[0]

    def click_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            self.redraw()

            if len(self.points) == 4:
                self.do_paste()

    def redraw(self):
        """Redraw the scene with current points."""
        display = self.scene_display.copy()
        colors = [
            (0, 255, 0),    # TL - green
            (255, 255, 0),  # TR - cyan
            (0, 0, 255),    # BR - red
            (255, 0, 255),  # BL - magenta
        ]
        labels = ["TL", "TR", "BR", "BL"]

        for i, (px, py) in enumerate(self.points):
            cv2.circle(display, (px, py), 6, colors[i], -1)
            cv2.circle(display, (px, py), 8, (255, 255, 255), 2)
            cv2.putText(display, labels[i], (px + 10, py - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            if i > 0:
                cv2.line(display, self.points[i-1], (px, py), (0, 255, 0), 2)
            if i == 3:
                cv2.line(display, (px, py), self.points[0], (0, 255, 0), 2)

        # Show instruction
        n = len(self.points)
        if n < 4:
            msg = f"Click {labels[n]} ({4 - n} remaining)  |  'r' = reset"
        else:
            msg = "'s' = save  |  'r' = reset  |  'q' = quit"
        cv2.putText(display, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display, msg, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 1)

        cv2.imshow("Interactive Billboard Pasting", display)

    def do_paste(self):
        """Warp content onto scene using the 4 clicked points."""
        # Convert display coordinates back to original image coordinates
        pts_orig = np.array(self.points, dtype=np.float32) / self.scale
        target_corners = pts_orig

        ch, cw = self.content_orig.shape[:2]
        src_corners = np.array([
            [0, 0], [cw - 1, 0], [cw - 1, ch - 1], [0, ch - 1]
        ], dtype=np.float32)

        # Compute homography
        H = cv2.getPerspectiveTransform(src_corners, target_corners)

        sh, sw = self.scene_orig.shape[:2]
        warped = cv2.warpPerspective(self.content_orig, H, (sw, sh))
        mask = cv2.warpPerspective(
            np.ones((ch, cw), dtype=np.uint8) * 255, H, (sw, sh))

        # Alpha blending at edges
        kernel = np.ones((5, 5), np.uint8)
        eroded = cv2.erode(mask, kernel, iterations=2)
        alpha = cv2.GaussianBlur(eroded, (15, 15), 7).astype(np.float32) / 255.0

        self.result = self.scene_orig.copy().astype(np.float32)
        warped_f = warped.astype(np.float32)
        scene_f = self.result.copy()

        for c in range(3):
            self.result[:, :, c] = (
                alpha * warped_f[:, :, c] +
                (1 - alpha) * scene_f[:, :, c]
            )
        self.result = self.result.astype(np.uint8)

        # Show result
        result_display, _ = resize_for_display(self.result)
        cv2.imshow("Result (press 's' to save, 'r' to redo)", result_display)

        # Also update main window
        display = self.scene_display.copy()
        cv2.putText(display, "DONE! 's'=save 'r'=reset 'q'=quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # Draw final quadrilateral
        pts_disp = np.array(self.points, dtype=np.int32)
        cv2.polylines(display, [pts_disp], True, (0, 255, 0), 2)
        cv2.imshow("Interactive Billboard Pasting", display)

        # Print the coordinates for use in scripts
        print(f"\nTarget corners (original scale):")
        for i, label in enumerate(["TL", "TR", "BR", "BL"]):
            print(f"  {label}: ({pts_orig[i][0]:.0f}, {pts_orig[i][1]:.0f})")
        print(f"\nAs numpy array:")
        print(f"  np.array({pts_orig.tolist()}, dtype=np.float32)")

    def save_result(self):
        """Save the current result."""
        if self.result is not None:
            os.makedirs("results", exist_ok=True)
            path = f"results/part3_{self.scene_name}_interactive.png"
            cv2.imwrite(path, self.result)
            print(f"Saved: {path}")

            # Also save with points marked
            scene_marked = self.scene_orig.copy()
            pts_orig = (np.array(self.points, dtype=np.float32) / self.scale).astype(int)
            for i in range(4):
                cv2.circle(scene_marked, tuple(pts_orig[i]), 10, (0, 0, 255), -1)
                cv2.line(scene_marked, tuple(pts_orig[i]),
                         tuple(pts_orig[(i+1) % 4]), (0, 0, 255), 3)
            path2 = f"results/part3_{self.scene_name}_corners.png"
            cv2.imwrite(path2, scene_marked)
            print(f"Saved: {path2}")

    def run(self):
        """Main loop."""
        cv2.namedWindow("Interactive Billboard Pasting", cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback("Interactive Billboard Pasting", self.click_handler)
        self.redraw()

        while True:
            key = cv2.waitKey(50) & 0xFF

            if key == ord('q') or key == 27:  # q or ESC
                break
            elif key == ord('r'):  # Reset
                self.points = []
                self.result = None
                cv2.destroyWindow("Result (press 's' to save, 'r' to redo)")
                self.redraw()
                print("Reset - click 4 new points.")
            elif key == ord('s'):  # Save
                self.save_result()

        cv2.destroyAllWindows()


def main():
    content_path = "images/bg3.jpg"
    if not os.path.exists(content_path):
        # Fallback: check for other extensions
        for ext in ['.png', '.jpeg', '.bmp']:
            alt = content_path.replace('.jpg', ext)
            if os.path.exists(alt):
                content_path = alt
                break
        else:
            print("ERROR: No content image found!")
            print("Please place your image as images/bg3.jpg")
            print("(This should be your personal photo or a celebrity photo)")
            sys.exit(1)

    print("=" * 50)
    print("Interactive Billboard Pasting Tool")
    print("=" * 50)
    print(f"Content image: {content_path}")
    print()

    scenes = []
    if os.path.exists("images/bg1.jpg"):
        scenes.append(("images/bg1.jpg", "Tablet (bg1)"))
    if os.path.exists("images/bg2.jpg"):
        scenes.append(("images/bg2.jpg", "Building H6 (bg2)"))

    if not scenes:
        print("No scene images found in images/ folder.")
        sys.exit(1)

    for scene_path, scene_name in scenes:
        print(f"\n--- Scene: {scene_name} ---")
        print("Click 4 corners: TL -> TR -> BR -> BL")
        print("Keys: 'r'=reset, 's'=save, 'q'=next/quit")
        print()

        paster = InteractivePaster(scene_path, content_path)
        paster.run()

    print("\nDone! Check results/ folder for saved images.")


if __name__ == "__main__":
    main()
