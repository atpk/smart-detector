"""
video_source.py
───────────────
Unified video source for:
  • Mac  — built-in FaceTime or any USB webcam (index 0)
  • Raspberry Pi — Camera Module v2/v3/HQ via OpenCV
  • File — any .mp4 / .avi for testing

Pi Camera Notes
───────────────
On Raspberry Pi OS Bullseye+ the default stack is libcamera.
OpenCV's VideoCapture(0) still works IF the v4l2 driver is active.

Enable it once:
    sudo modprobe bcm2835-v4l2          # older Pi OS (Buster / Bullseye)
    # or add  dtoverlay=vc4-fkms-v3d   to /boot/config.txt and reboot

If you get "Cannot open camera 0" on Pi, run:
    v4l2-ctl --list-devices
and set CAMERA_INDEX in config.py to the correct /dev/videoN index.
"""

import cv2
import time
import platform
import config


class VideoSource:
    """
    Yields (frame_index, frame_bgr) tuples from a live camera or video file.

    Usage
    -----
    source = VideoSource()
    for frame_idx, frame in source.frames():
        process(frame)
    """

    def __init__(self):
        self._path  = config.VIDEO_PATH
        self._index = config.CAMERA_INDEX
        self._loop  = config.LOOP_VIDEO
        self._cap   = None
        self._frame_delay = 1.0 / config.PLAYBACK_FPS

    # ── Public ────────────────────────────────────────────────────────────

    def frames(self):
        """Generator: yields (frame_index, frame_bgr) tuples indefinitely."""
        self._open()
        frame_idx = 0
        try:
            while True:
                t_start = time.time()
                ok, frame = self._cap.read()

                if not ok:
                    if self._loop and self._path:
                        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = self._cap.read()
                        if not ok:
                            print("[VideoSource] Rewind failed – stopping.")
                            break
                    else:
                        # Camera returned empty frame – transient, retry once
                        time.sleep(0.05)
                        ok, frame = self._cap.read()
                        if not ok:
                            print("[VideoSource] Camera feed lost.")
                            break

                yield frame_idx, frame
                frame_idx += 1

                # Throttle only for file playback; live camera self-paces
                if self._path:
                    elapsed = time.time() - t_start
                    wait    = self._frame_delay - elapsed
                    if wait > 0:
                        time.sleep(wait)

        except KeyboardInterrupt:
            pass
        finally:
            self.release()

    def release(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()

    # ── Private ───────────────────────────────────────────────────────────

    def _open(self):
        if self._path:
            # ── File mode ────────────────────────────────────────────────
            self._cap = cv2.VideoCapture(self._path)
            if not self._cap.isOpened():
                raise RuntimeError(
                    f"[VideoSource] Cannot open file: '{self._path}'"
                )
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                self._frame_delay = 1.0 / fps
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"[VideoSource] Opened file: {self._path}  ({w}×{h} @ {fps:.1f} fps)")

        else:
            # ── Live camera mode ─────────────────────────────────────────
            self._cap = self._open_camera()
            if not self._cap.isOpened():
                raise RuntimeError(
                    "[VideoSource] Cannot open camera.\n"
                    "  → Check CAMERA_INDEX in config.py\n"
                    "  → On Pi: run 'v4l2-ctl --list-devices' to find the right index\n"
                    "  → On Mac: grant Terminal/Python camera permission in\n"
                    "     System Preferences → Privacy & Security → Camera"
                )

            # Request 640×480 @ 30 fps — works on both Mac and Pi Camera Module
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self._cap.set(cv2.CAP_PROP_FPS,          30)

            # Mac: disable auto-focus hunt (reduces initial latency)
            if platform.system() == "Darwin":
                self._cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

            w   = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h   = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self._cap.get(cv2.CAP_PROP_FPS)
            sys_name = platform.system()
            print(
                f"[VideoSource] Opened camera index={self._index} "
                f"on {sys_name}  ({w}×{h} @ {fps:.1f} fps)"
            )

    def _open_camera(self):
        """
        Try configured index with default backend first.
        If that fails on Pi, retry with V4L2 backend explicitly.
        """
        cap = cv2.VideoCapture(self._index)
        if cap.isOpened():
            return cap

        # Fallback: force V4L2 backend (useful on some Pi OS builds)
        cap.release()
        cap = cv2.VideoCapture(self._index, cv2.CAP_V4L2)
        return cap
