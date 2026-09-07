"""Mandatory integration test: moving synthetic subject with dynamic crop tracking and FFmpeg rendering."""

from pathlib import Path
import subprocess
import cv2
import numpy as np

from freecher_worker.crop.detector import SubjectDetector
from freecher_worker.crop.expression import build_ffmpeg_crop_x_expression
from freecher_worker.crop.models import DetectedSubject
from freecher_worker.crop.tracker import generate_crop_trajectory


class MovingBoxDetector(SubjectDetector):
    """Detector for synthetic moving white square on black background."""

    def detect(self, frame_bgr: np.ndarray, timestamp: float) -> list[DetectedSubject]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours((gray > 200).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        subjects = []
        for c in contours:
            x, y, w, h = cv2.boundingRect(c)
            if w >= 30 and h >= 30:
                subjects.append(
                    DetectedSubject(
                        box=(int(x), int(y), int(w), int(h)),
                        confidence=1.0,
                        subject_type="face",
                        area=float(w * h),
                        center_x=float(x + w / 2.0),
                        center_y=float(y + h / 2.0),
                    )
                )
        return subjects


def test_moving_synthetic_subject_dynamic_crop_execution(tmp_path):
    """Verify tracker captures left-to-right movement (300 -> 1050) and FFmpeg moves crop window."""
    video_path = tmp_path / "moving_synthetic.mp4"
    out_crop_mp4 = tmp_path / "dynamic_cropped_9_16.mp4"

    # 1. Create synthetic 1920x1080 3-second video (60 frames @ 20 FPS)
    # The subject moves from x=300 to x=1050
    w_src, h_src = 1920, 1080
    fps = 20
    duration = 3.0
    total_frames = int(fps * duration)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (w_src, h_src))

    for i in range(total_frames):
        t = i / float(fps)
        frame = np.zeros((h_src, w_src, 3), dtype=np.uint8)

        # Subject moves from 300 to 1050
        center_x = int(300.0 + (t / duration) * (1050.0 - 300.0))
        center_y = 540
        box_size = 100

        # Draw white box (simulate face)
        x1 = center_x - box_size // 2
        y1 = center_y - box_size // 2
        cv2.rectangle(frame, (x1, y1), (x1 + box_size, y1 + box_size), (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    assert video_path.is_file()

    # 2. Run tracker with analysis_fps=2.0
    detector = MovingBoxDetector()
    trajectory = generate_crop_trajectory(
        video_path=video_path,
        start_seconds=0.0,
        duration_seconds=duration,
        detector=detector,
        analysis_fps=2.0,
        deadzone_ratio=0.01,
        max_velocity_px_per_sec=400.0,
    )

    pts = trajectory.points
    assert len(pts) >= 4

    # Verify trajectory coordinates moved monotonically from left to right
    first_cx = pts[0].center_x
    last_cx = pts[-1].center_x
    assert 250 <= first_cx <= 350
    assert 950 <= last_cx <= 1100
    assert pts[-1].crop_x > pts[0].crop_x

    # 3. Build FFmpeg crop expression
    crop_x_expr = build_ffmpeg_crop_x_expression(trajectory)
    assert "if(lte(t" in crop_x_expr

    # 4. Render vertical crop via FFmpeg using the dynamic expression
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-y",
        "-i", str(video_path),
        "-vf", f"crop={trajectory.crop_w}:{trajectory.crop_h}:{crop_x_expr}:0,scale=1080:1920",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(out_crop_mp4),
    ]

    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=False)
    assert res.returncode == 0, f"FFmpeg crop failed: {res.stderr}"
    assert out_crop_mp4.is_file()

    # 5. Verify the rendered video actually contains the subject throughout the motion
    # In both the first frame (t=0.1) and final frame (t=2.8), the subject should be visible inside the 1080x1920 frame!
    cap_out = cv2.VideoCapture(str(out_crop_mp4))
    assert cap_out.isOpened()

    # Frame 0
    ret, frame_start = cap_out.read()
    assert ret
    assert frame_start.shape == (1920, 1080, 3)
    # Check that white box is present in frame_start
    assert np.max(frame_start) > 200

    # Seek towards end (frame 50 / 60)
    cap_out.set(cv2.CAP_PROP_POS_FRAMES, 50)
    ret, frame_end = cap_out.read()
    assert ret
    # White box should still be present because camera dynamically tracked it!
    assert np.max(frame_end) > 200

    cap_out.release()
