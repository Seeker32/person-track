"""Camera abstraction for video capture (webcam or RTSP)."""

import logging
import time
from collections.abc import Generator

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class Camera:
    """Wraps cv2.VideoCapture for webcam indices and RTSP URLs.

    Args:
        name: Human-readable camera name.
        source: Integer for webcam index or string for RTSP URL / file path.
        max_retries: Maximum reconnection attempts for RTSP streams.
        retry_delay: Seconds to wait between reconnection attempts.
    """

    def __init__(
        self,
        name: str,
        source: int | str,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> None:
        self.name = name
        self.source = source
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._cap: cv2.VideoCapture | None = None
        self._open()

    def _open(self) -> None:
        """Open the video capture device."""
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise ValueError(
                f"Camera '{self.name}': failed to open source '{self.source}'"
            )

    @property
    def video_info(self) -> dict[str, float]:
        """Return video metadata: width, height, fps."""
        if self._cap is None:
            return {"width": 0, "height": 0, "fps": 0}
        return {
            "width": self._cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

    def frames(self) -> Generator[tuple[int, np.ndarray], None, None]:
        """Yield (frame_id, frame) from the camera.

        For RTSP streams, attempts reconnection on read failure.
        """
        frame_id = 0
        consecutive_failures = 0

        while True:
            if self._cap is None:
                break

            ret, frame = self._cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.warning(
                    "Camera '%s': failed to read frame (attempt %d)",
                    self.name,
                    consecutive_failures,
                )
                if consecutive_failures >= self.max_retries:
                    if not self._reconnect():
                        logger.error(
                            "Camera '%s': reconnection failed, stopping.",
                            self.name,
                        )
                        break
                    consecutive_failures = 0
                else:
                    time.sleep(self.retry_delay)
                continue

            consecutive_failures = 0
            yield frame_id, frame
            frame_id += 1

    def _reconnect(self) -> bool:
        """Attempt to reconnect the camera. Returns True on success."""
        logger.info("Camera '%s': attempting reconnection...", self.name)
        self.release()
        time.sleep(self.retry_delay)
        try:
            self._open()
            logger.info("Camera '%s': reconnected successfully.", self.name)
            return True
        except ValueError:
            return False

    def release(self) -> None:
        """Release the capture device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
