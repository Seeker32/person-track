"""YOLOv6 detector wrapper using Ultralytics."""

import logging
from pathlib import Path

import numpy as np
import supervision as sv
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class Detector:
    """Loads a YOLO model and runs person detection on frames.

    Args:
        model_path: Path to the .pt model file.
        conf_threshold: Minimum confidence for detections.
        iou_threshold: NMS IoU threshold.
        device: 'cpu' or 'cuda:0'.
        classes: List of COCO class IDs to detect (0 = person).
    """

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.7,
        device: str = "cpu",
        classes: list[int] | None = None,
    ) -> None:
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes if classes is not None else [0]

        model_file = Path(model_path)
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading model from %s on %s...", model_path, device)
        self._model = YOLO(str(model_file))
        # Warm-up: run a dummy inference to load model into memory
        self._model.predict(
            source=np.zeros((640, 640, 3), dtype=np.uint8),
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            classes=self.classes,
            verbose=False,
        )
        logger.info("Model loaded and ready.")

    def detect(self, frame: np.ndarray) -> sv.Detections:
        """Run detection on a single frame.

        Returns:
            sv.Detections filtered to the configured person class(es).
        """
        results = self._model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )

        if len(results) == 0 or results[0].boxes is None:
            return sv.Detections.empty()

        detections = sv.Detections.from_ultralytics(results[0])

        # Filter to configured classes (belt-and-suspenders)
        if len(detections) > 0:
            mask = np.isin(detections.class_id, self.classes)
            detections = detections[mask]

        return detections
