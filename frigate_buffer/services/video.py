import os
import subprocess
import logging
from typing import Optional

logger = logging.getLogger('frigate-buffer')


class VideoService:
    """Handles video processing tasks like transcoding and GIF generation."""

    def __init__(self, ffmpeg_timeout: int = 60):
        self.ffmpeg_timeout = ffmpeg_timeout
        logger.debug(f"VideoService initialized with FFmpeg timeout: {ffmpeg_timeout}s")

    def transcode_clip_to_h264(self, event_id: str, temp_path: str, final_path: str) -> bool:
        """Transcode clip_original.mp4 to H.264 clip.mp4. Removes temp on success."""
        process = None
        try:
            command = [
                "ffmpeg", "-y",
                "-i", temp_path,
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-c:a", "aac",
                "-movflags", "+faststart",
                final_path,
            ]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=self.ffmpeg_timeout)
            if process.returncode == 0:
                os.remove(temp_path)
                logger.info(f"Transcoded clip for {event_id}")
                return True
            logger.error(f"FFmpeg error for {event_id}: {stderr.decode()[:500]}")
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return True
        except subprocess.TimeoutExpired:
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return True
        except Exception as e:
            logger.exception(f"Transcode failed for {event_id}: {e}")
            self._terminate_process_gracefully(process, event_id)
            if os.path.exists(temp_path):
                os.rename(temp_path, final_path)
            return True

    def generate_gif_from_clip(self, clip_path: str, output_path: str,
                               fps: int = 5, duration_sec: float = 5.0) -> bool:
        """Generate animated GIF from video clip using FFmpeg. Returns True on success."""
        try:
            scale = "320:-1"
            cmd = [
                "ffmpeg", "-y", "-i", clip_path,
                "-vf", f"fps={fps},scale={scale}",
                "-t", str(duration_sec),
                output_path
            ]
            proc = subprocess.run(cmd, capture_output=True, timeout=self.ffmpeg_timeout)
            if proc.returncode == 0 and os.path.exists(output_path):
                logger.info(f"Generated GIF from {clip_path}")
                return True
        except FileNotFoundError:
            logger.warning("GIF generation failed: 'ffmpeg' executable not found")
        except subprocess.CalledProcessError as e:
            logger.warning(f"GIF generation failed (subprocess error): {e}")
        except Exception as e:
            logger.warning(f"GIF generation failed: {e}")
        return False

    def _terminate_process_gracefully(self, process, event_id: str, timeout: float = 5.0):
        """Gracefully terminate a process: SIGTERM first, then SIGKILL if needed."""
        if process is None or process.poll() is not None:
            return  # Already dead

        logger.debug(f"Sending SIGTERM to FFmpeg for {event_id}")
        try:
            process.terminate()  # SIGTERM - allows graceful shutdown
        except OSError:
            return  # Process already gone

        try:
            process.wait(timeout=timeout)  # Wait for graceful exit
            logger.debug(f"FFmpeg for {event_id} terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning(f"FFmpeg for {event_id} didn't respond to SIGTERM, sending SIGKILL")
            try:
                process.kill()  # SIGKILL - force kill
                process.wait()  # Reap zombie
            except OSError:
                pass  # Process already gone
