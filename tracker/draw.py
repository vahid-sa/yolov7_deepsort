import cv2
import random
import numpy as np
from collections import deque

from .deep_sort.track import Track
from .deep_sort.detection import Detection


class Drawer:
    """Detection and Tracking drawer."""

    def __init__(self) -> None:
        """Class constructor."""
        random.seed(3)
        self._COLORS = np.random.randint(0, 255, size=(200, 3), dtype=np.uint8)
        self._pts = [deque(maxlen=30) for _ in range(9999)]

    def __call__(
        self, image: np.ndarray, detections: list, tracks: list,
    ) -> np.ndarray:
        """Draws objects on the image.

        Args:
            image (np.ndarray): IO BGR image
            detections (list): detected objects from deep_sort
            tracks (list): tracked objects

        Returns:
            np.ndarray: Image specified objects on.
        """
        img: np.ndarray = image.copy()
        i = 0
        for j in range(len(detections)):
            det: Detection = detections[j]
            bbox = det.to_tlbr()
            cv2.rectangle(img,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            color = (255, 255, 255)
            # cv2.putText(image, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)
        indexIDs = []
        for j in range(len(tracks)):
            track: Track = tracks[j]
            if (not track.is_confirmed()) or (track.time_since_update > 1):
                continue
            indexIDs.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in self._COLORS[indexIDs[i] % len(self._COLORS)]]
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(img,str(track.track_id),(int(bbox[0]), int(bbox[1] -25)),0, 5e-3 * 150, (color),1)
            cv2.putText(img, track.class_name,(int(bbox[0]), int(bbox[1] -10)),0, 5e-3 * 150, (color),1)
            center = (
                int((bbox[0] + bbox[2]) / 2.0),
                int((bbox[1] + bbox[3]) / 2.0),
            )
            self._pts[track.track_id].append(center)
            thickness = 5
            cv2.circle(img, center, 1, color, thickness)

            for k in range(1, len(self._pts[track.track_id])):
                if (self._pts[track.track_id][k - 1] is None) or (self._pts[track.track_id][k] is None):
                    continue
                thickness = int(np.sqrt(64 / float(k + 1)) * 2)
                cv2.line(
                    img,
                    self._pts[track.track_id][k - 1],
                    self._pts[track.track_id][k],
                    color,
                    thickness,
                )
            i += 1
        return img
