import numpy as np
from typing import Tuple

from .deep_sort import tracker
from .deep_sort.detection import Detection
from .deep_sort import nn_matching
from . import generate_detections as gdet


class Tracker:
    """Tracks predicted objects using deep_sort"""

    def __init__(self, protbuf_path: str) -> None:
        """Class constructor.

        Args:
            protbuf_path (str): Path to .protbuf file for Person ReID.
        """
        self._encoder = gdet.create_box_encoder(protbuf_path, batch_size=1)
        self._tracker = tracker.Tracker(
            metric=nn_matching.NearestNeighborDistanceMetric(
                metric="cosine",
                matching_threshold=0.3,  # max_cosine_distance
                budget=None,  # nn_bdget
            ),
        )

    def __call__(
        self,
        frame: np.ndarray,
        predictions: Tuple[np.ndarray, list, list],
    ) -> Tuple[list, list]:
        """Run tracking on predicted objects for the current image.

        Args:
            frame (Tuple[np.ndarray, list, list]): IO BGR image
            predictions (np.ndarray): Predicted objects by object detector.

        Returns:
            Tuple[list, list]: deep_sort detections, tracks
        """
        # classes, confidences, bboxes = predictions
        bboxes: np.ndarray = predictions[0]
        confidences: list = predictions[1]
        classes: list = predictions[2]
        #  assert len(classes) == len(confidences), len(bboxes)
        if len(classes) > 0:
            adjusted_boxes = Tracker._ltrb2ltwh(
                bboxes=bboxes,
            )
            adjusted_boxes = adjusted_boxes.tolist()
            features: list = self._encoder(frame, adjusted_boxes)
            detections = [
                Detection(
                    bbox, 1.0, feature,
                ) for bbox, feature in zip(
                    adjusted_boxes, features,
                )
            ]
        else:
            detections = []
            classes = []
        self._tracker.predict()
        self._tracker.update(detections, classes)
        return detections, self._tracker.tracks

    @staticmethod
    def _ltrb2ltwh(bboxes: np.ndarray) -> np.ndarray:
        """converts (l, t, r, b) to (l, t, w, h).

        Args:
            bboxes (np.ndarray): left, top, right, bottom; shape: (N, 4)

        Returns:
            np.ndarray: left, top, width, height; shape: (N, 4)
        """
        left = bboxes[:, 0]
        top = bboxes[:, 1]
        right = bboxes[:, 2]
        bottom = bboxes[:, 3]
        width = right - left
        height = bottom - top
        ltwh = np.concatenate(
            [
                left[:, np.newaxis],
                top[:, np.newaxis],
                width[:, np.newaxis],
                height[:, np.newaxis],
            ],
            axis=1,
        )
        return ltwh
