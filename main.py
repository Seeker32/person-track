#!/usr/bin/env python3
"""Person tracking system: YOLOv6 detection + Bot-SORT tracking.

Usage:
    uv run python main.py --config config.yaml
"""

import argparse
import logging
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent / "src"))

from camera import Camera
from detector import Detector
from tracker import Tracker
from visualizer import Visualizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FPS Counter
# ---------------------------------------------------------------------------

class FPSCounter:
    """Rolling-window FPS tracker."""

    def __init__(self, window: int = 30) -> None:
        self._timestamps: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        """Record a frame event."""
        self._timestamps.append(time.monotonic())

    @property
    def fps(self) -> float:
        """Current rolling-average FPS."""
        if len(self._timestamps) < 2:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        return (len(self._timestamps) - 1) / elapsed if elapsed > 0 else 0.0


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load and validate YAML configuration."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError("Config file is empty.")

    required_sections = ["cameras", "detection", "tracking", "visualization"]
    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required config section: '{section}'"
            )

    if not config["cameras"]:
        raise ValueError("At least one camera must be configured.")

    return config


# ---------------------------------------------------------------------------
# FPS overlay
# ---------------------------------------------------------------------------

def draw_fps(frame: np.ndarray, fps: float) -> None:
    """Draw FPS counter in the top-left corner of the frame."""
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        frame,
        text,
        org=(10, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 255, 0),
        thickness=2,
        lineType=cv2.LINE_AA,
    )


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Person tracking with YOLOv6 + Bot-SORT"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to configuration YAML file (default: config.yaml)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Load config ---
    try:
        config = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        sys.exit(1)

    det_cfg = config["detection"]
    trk_cfg = config["tracking"]
    vis_cfg = config["visualization"]
    out_cfg = config.get("output", {})

    # --- Initialize components ---
    detector = Detector(
        model_path=det_cfg["model_path"],
        conf_threshold=det_cfg.get("conf_threshold", 0.25),
        iou_threshold=det_cfg.get("iou_threshold", 0.7),
        device=det_cfg.get("device", "cpu"),
        classes=det_cfg.get("classes", [0]),
    )

    tracker = Tracker(
        lost_track_buffer=trk_cfg.get("lost_track_buffer", 30),
        frame_rate=trk_cfg.get("frame_rate", 30.0),
        track_activation_threshold=trk_cfg.get(
            "track_activation_threshold", 0.7
        ),
        minimum_consecutive_frames=trk_cfg.get(
            "minimum_consecutive_frames", 2
        ),
        minimum_iou_threshold_first_assoc=trk_cfg.get(
            "minimum_iou_threshold_first_assoc", 0.2
        ),
        minimum_iou_threshold_second_assoc=trk_cfg.get(
            "minimum_iou_threshold_second_assoc", 0.5
        ),
        minimum_iou_threshold_unconfirmed_assoc=trk_cfg.get(
            "minimum_iou_threshold_unconfirmed_assoc", 0.3
        ),
        high_conf_det_threshold=trk_cfg.get("high_conf_det_threshold", 0.6),
        enable_cmc=trk_cfg.get("enable_cmc", True),
        cmc_method=trk_cfg.get("cmc_method", "sparseOptFlow"),
        cmc_downscale=trk_cfg.get("cmc_downscale", 2),
    )

    visualizer = Visualizer(
        show_traces=vis_cfg.get("show_traces", True),
        trace_length=vis_cfg.get("trace_length", 60),
        box_thickness=vis_cfg.get("box_thickness", 2),
        label_text_scale=vis_cfg.get("label_text_scale", 0.6),
        label_text_thickness=vis_cfg.get("label_text_thickness", 1),
    )

    display_width = vis_cfg.get("display_width")
    display_height = vis_cfg.get("display_height")

    # --- Video writer (optional) ---
    video_writer: cv2.VideoWriter | None = None
    if out_cfg.get("save_video"):
        out_path = Path(out_cfg.get("video_output_path", "output/tracked.mp4"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # We'll initialise the writer after we know the frame size

    logger.info("Starting tracking pipeline...")

    for cam_cfg in config["cameras"]:
        cam_name = cam_cfg["name"]
        cam_source = cam_cfg["source"]
        logger.info("Opening camera '%s' (source: %s)...", cam_name, cam_source)

        camera = Camera(name=cam_name, source=cam_source)

        info = camera.video_info
        logger.info(
            "Camera '%s': %dx%d @ %.1f fps",
            cam_name,
            int(info["width"]),
            int(info["height"]),
            info["fps"],
        )

        # Initialise video writer on first frame if saving
        if video_writer is None and out_cfg.get("save_video"):
            out_w = display_width or int(info["width"])
            out_h = display_height or int(info["height"])
            out_fps = out_cfg.get("save_fps") or info["fps"] or 30.0
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out_path = out_cfg.get("video_output_path", "output/tracked.mp4")
            video_writer = cv2.VideoWriter(
                out_path, fourcc, out_fps, (out_w, out_h)
            )
            logger.info("Saving output video to %s", out_path)

        fps_counter = FPSCounter()
        window_name = f"Person Track - {cam_name}"

        try:
            for frame_id, frame in camera.frames():
                fps_counter.tick()

                # --- Detect ---
                detections = detector.detect(frame)

                # --- Track ---
                tracked = tracker.update(detections, frame)

                # --- Annotate ---
                annotated = visualizer.annotate(frame, tracked)

                # --- Resize for display ---
                if display_width and display_height:
                    annotated = cv2.resize(
                        annotated, (display_width, display_height)
                    )

                # --- FPS overlay ---
                if frame_id % 10 == 0:  # update FPS text every 10 frames
                    draw_fps(annotated, fps_counter.fps)

                # --- Display ---
                if vis_cfg.get("display", True):
                    cv2.imshow(window_name, annotated)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # q or ESC
                        logger.info("User requested exit.")
                        return

                # --- Save ---
                if video_writer is not None:
                    video_writer.write(annotated)

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            camera.release()

    cv2.destroyAllWindows()
    if video_writer is not None:
        video_writer.release()
    logger.info("Tracking pipeline finished.")


if __name__ == "__main__":
    main()
