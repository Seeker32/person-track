"""Bot-SORT tracker wrapper using the trackers package."""

import logging

import numpy as np
import supervision as sv
from trackers import BoTSORTTracker

logger = logging.getLogger(__name__)


class Tracker:
    """Wraps BoTSORTTracker for person tracking.

    Args:
        lost_track_buffer: Frames before a lost track is removed.
        frame_rate: Video frame rate for motion modeling.
        track_activation_threshold: Minimum detection confidence to start a track.
        minimum_consecutive_frames: Frames needed to confirm a new track.
        minimum_iou_threshold_first_assoc: IoU threshold for first association.
        minimum_iou_threshold_second_assoc: IoU threshold for second association.
        minimum_iou_threshold_unconfirmed_assoc: IoU threshold for unconfirmed tracks.
        high_conf_det_threshold: Confidence above which detections skip first association.
        enable_cmc: Enable camera motion compensation.
        cmc_method: CMC method ('sparseOptFlow', 'orb', 'sift', 'ecc').
        cmc_downscale: Downscale factor for CMC computation.
    """

    def __init__(
        self,
        lost_track_buffer: int = 30,
        frame_rate: float = 30.0,
        track_activation_threshold: float = 0.7,
        minimum_consecutive_frames: int = 2,
        minimum_iou_threshold_first_assoc: float = 0.2,
        minimum_iou_threshold_second_assoc: float = 0.5,
        minimum_iou_threshold_unconfirmed_assoc: float = 0.3,
        high_conf_det_threshold: float = 0.6,
        enable_cmc: bool = True,
        cmc_method: str = "sparseOptFlow",
        cmc_downscale: int = 2,
    ) -> None:
        self._tracker = BoTSORTTracker(
            lost_track_buffer=lost_track_buffer,
            frame_rate=frame_rate,
            track_activation_threshold=track_activation_threshold,
            minimum_consecutive_frames=minimum_consecutive_frames,
            minimum_iou_threshold_first_assoc=minimum_iou_threshold_first_assoc,
            minimum_iou_threshold_second_assoc=minimum_iou_threshold_second_assoc,
            minimum_iou_threshold_unconfirmed_assoc=minimum_iou_threshold_unconfirmed_assoc,
            high_conf_det_threshold=high_conf_det_threshold,
            enable_cmc=enable_cmc,
            cmc_method=cmc_method,  # type: ignore[arg-type]
            cmc_downscale=cmc_downscale,
        )

    def update(
        self, detections: sv.Detections, frame: np.ndarray
    ) -> sv.Detections:
        """Update the tracker with new detections.

        Args:
            detections: Current frame detections (sv.Detections).
            frame: The raw frame (needed for CMC).

        Returns:
            sv.Detections with tracker_id field added.
        """
        if len(detections) == 0:
            detections = sv.Detections.empty()

        tracked = self._tracker.update(detections=detections, frame=frame)
        return tracked

    def reset(self) -> None:
        """Reset the tracker state (e.g., after camera reconnection)."""
        self._tracker.reset()
