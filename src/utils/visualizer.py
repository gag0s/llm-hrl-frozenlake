import time
import cv2
import numpy as np


class Visualizer:
    """
    Handles rendering overlays for agent evaluation:
      - translucent red box for the current subgoal bin (based on goal_idx)
      - action indicator (0=LEFT, 1=DOWN, 2=RIGHT, 3=UP) drawn as an arrow in the top-right
    """

    def __init__(self, env, show: bool):
        self.env = env
        self.window_name = "Agent Evaluation"
        self.opacity = 0.2
        self.show_window = show
        self.grid_size = 8  # 8x8 map

        # Badge config
        self.badge_size = 72       # square size in pixels
        self.badge_margin = 12     # distance from edges
        self.badge_alpha = 0.45    # background translucency
        self.arrow_thickness = 4
        self.arrow_tip_len = 0.35  # relative length of arrow head

        if self.show_window:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 640, 480)

    def close(self):
        if self.show_window:
            cv2.destroyWindow(self.window_name)

    def annotate(self, frame: np.ndarray, goal_idx: int, action: int) -> np.ndarray:
        """
        Returns a copy of the frame with:
          - a translucent red overlay over the cell corresponding to goal_idx (0–15 for a 4x4 grid)
          - a top-right arrow badge indicating the current action if provided
        """
        annotated = frame.copy()

        # --- draw subgoal cell overlay ---
        h, w = annotated.shape[:2]
        cell_w = w // self.grid_size
        cell_h = h // self.grid_size

        row = int(goal_idx) // self.grid_size
        col = int(goal_idx) % self.grid_size

        x1, y1 = col * cell_w, row * cell_h
        x2, y2 = x1 + cell_w, y1 + cell_h

        overlay = annotated.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)  # red fill
        annotated = cv2.addWeighted(overlay, self.opacity, annotated, 1 - self.opacity, 0)

        # --- draw action indicator badge (if not neutral) ---
        if action is not None:
            annotated = self._draw_action_badge(annotated, int(action))

        return annotated

    def _draw_action_badge(self, img: np.ndarray, action: int) -> np.ndarray:
        """
        Draws a semi-transparent square badge in the top-right with an arrow
        indicating the action: 0=left,1=down,2=right,3=up
        """
        h, w = img.shape[:2]
        size = min(self.badge_size, max(40, int(0.14 * min(h, w))))  # scale a bit with frame
        margin = self.badge_margin

        # Badge rectangle coords (top-right)
        x2 = w - margin
        y1 = margin
        x1 = x2 - size
        y2 = y1 + size

        # Semi-transparent dark background
        roi = img[y1:y2, x1:x2].copy()
        bg = roi.copy()
        cv2.rectangle(bg, (0, 0), (size, size), (0, 0, 0), -1)  # black
        img[y1:y2, x1:x2] = cv2.addWeighted(bg, self.badge_alpha, roi, 1 - self.badge_alpha, 0)

        # Compute arrow start/end inside the badge
        cx = x1 + size // 2
        cy = y1 + size // 2
        pad = max(8, size // 5)
        # Arrow color: green-ish
        color = (60, 200, 60)

        if action == 0:  # left
            start = (x2 - pad, cy)
            end   = (x1 + pad, cy)
        elif action == 1:  # down
            start = (cx, y1 + pad)
            end   = (cx, y2 - pad)
        elif action == 2:  # right
            start = (x1 + pad, cy)
            end   = (x2 - pad, cy)
        elif action == 3:  # up
            start = (cx, y2 - pad)
            end   = (cx, y1 + pad)
        else:
            # unknown action -> draw a question mark
            cv2.putText(img, "?", (x1 + size // 3, y1 + int(size * 0.68)),
                        cv2.FONT_HERSHEY_SIMPLEX, max(0.5, size / 80), (255, 255, 255), 2, cv2.LINE_AA)
            return img

        cv2.arrowedLine(img, start, end, color, thickness=self.arrow_thickness,
                        tipLength=self.arrow_tip_len)

        # Optional tiny label
        labels = {0: "L", 1: "D", 2: "R", 3: "U"}
        cv2.putText(img, labels.get(action, ""),
                    (x1 + 6, y2 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, max(0.4, size / 110),
                    (220, 220, 220), 1, cv2.LINE_AA)

        return img

    def show(self, frame: np.ndarray, delay_ms: int = 30) -> bool:
        """
        Displays the frame in a window.
        Returns True if the user presses 'q' to quit.
        """
        if not self.show_window:
            return False

        time.sleep(0.5)
        cv2.imshow(self.window_name, frame)
        return (cv2.waitKey(delay_ms) & 0xFF) == ord('q')
