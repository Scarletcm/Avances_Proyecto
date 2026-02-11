import cv2
import numpy as np


class OpticalFlowService:
    def __init__(self):
        self.prev_gray = None

    def process(self, frame):
        if frame is None:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return {"motion_level": 0.0}

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        motion_level = float(np.mean(mag))

        self.prev_gray = gray
        return {"motion_level": motion_level}
