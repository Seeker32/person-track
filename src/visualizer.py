"""Visualization helpers using supervision annotators."""

import numpy as np
import supervision as sv


class Visualizer:
    """Annotates frames with bounding boxes, track IDs, and motion traces.

    Args:
        show_traces: Whether to draw movement traces.
        trace_length: Number of frames a trace persists.
        box_thickness: Thickness of bounding box lines.
        label_text_scale: Font scale for ID labels.
        label_text_thickness: Font thickness for ID labels.
    """

    def __init__(
        self,
        show_traces: bool = True,
        trace_length: int = 60,
        box_thickness: int = 2,
        label_text_scale: float = 0.6,
        label_text_thickness: int = 1,
    ) -> None:
        self.show_traces = show_traces

        self.box_annotator = sv.BoxAnnotator(
            color=sv.ColorPalette.DEFAULT,
            thickness=box_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.label_annotator = sv.LabelAnnotator(
            color=sv.ColorPalette.DEFAULT,
            text_color=sv.Color.from_hex("#FFFFFF"),
            text_scale=label_text_scale,
            text_thickness=label_text_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )
        self.trace_annotator = sv.TraceAnnotator(
            color=sv.ColorPalette.DEFAULT,
            position=sv.Position.BOTTOM_CENTER,
            trace_length=trace_length,
            thickness=box_thickness,
            color_lookup=sv.ColorLookup.TRACK,
        )

    def annotate(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        """Draw annotations on the frame.

        Args:
            frame: The raw BGR frame.
            detections: Tracked detections with tracker_id.

        Returns:
            Annotated frame (new copy, original unchanged).
        """
        annotated = frame.copy()

        # Only visualize confirmed tracks
        if len(detections) == 0:
            return annotated

        if hasattr(detections, "tracker_id") and detections.tracker_id is not None:
            confirmed_mask = detections.tracker_id >= 0
            detections = detections[confirmed_mask]

        if len(detections) == 0:
            return annotated

        # Traces first (behind boxes)
        if self.show_traces:
            annotated = self.trace_annotator.annotate(
                scene=annotated, detections=detections
            )

        # Bounding boxes
        annotated = self.box_annotator.annotate(
            scene=annotated, detections=detections
        )

        # Track ID labels
        if (
            hasattr(detections, "tracker_id")
            and detections.tracker_id is not None
        ):
            labels = [f"ID:{tid}" for tid in detections.tracker_id]
        else:
            labels = [f"#{i}" for i in range(len(detections))]

        annotated = self.label_annotator.annotate(
            scene=annotated, detections=detections, labels=labels
        )

        return annotated
