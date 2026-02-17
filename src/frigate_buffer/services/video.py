import os
import subprocess
import logging

import ffmpegcv

logger = logging.getLogger('frigate-buffer')


class VideoService:
    """Handles video processing tasks like transcoding and GIF generation."""

    DEFAULT_FFMPEG_TIMEOUT = 60

    def __init__(self, ffmpeg_timeout: int = DEFAULT_FFMPEG_TIMEOUT):
        self.ffmpeg_timeout = ffmpeg_timeout
        logger.debug(f"VideoService initialized with FFmpeg timeout: {ffmpeg_timeout}s")

    def transcode_clip_to_h264(self, event_id: str, temp_path: str, final_path: str) -> bool:
        """Transcode clip_original.mp4 to H.264 clip.mp4 (NVDEC decode, NVENC encode when GPU available).
        Removes temp on success. Falls back to libx264 if GPU path fails."""
        try:
            if self._transcode_clip_nvenc(event_id, temp_path, final_path):
                return True
        except Exception as e:
            logger.warning(
                "GPU unavailable for transcode (%s), falling back to libx264 (CPU). Reason: %s: %s",
                event_id, type(e).__name__, e,
            )
        return self._transcode_clip_libx264(event_id, temp_path, final_path)

    def _transcode_clip_nvenc(self, event_id: str, temp_path: str, final_path: str) -> bool:
        """Decode with VideoCaptureNV (NVDEC), encode with VideoWriterNV (h264_nvenc), mux audio via ffmpeg."""
        cap = ffmpegcv.VideoCaptureNV(temp_path)
        if not cap.isOpened():
            raise RuntimeError("VideoCaptureNV could not open input")
        try:
            fps = getattr(cap, "fps", None) or 30.0
            ret, first_frame = cap.read()
            if not ret or first_frame is None:
                raise RuntimeError("No frames in input")
            h, w = first_frame.shape[:2]
            dirname = os.path.dirname(final_path)
            video_only_path = os.path.join(dirname, "clip_nvenc_tmp.mp4")
            try:
                vidout = ffmpegcv.VideoWriterNV(
                    video_only_path,
                    "h264_nvenc",
                    fps,
                    pix_fmt="bgr24",
                )
                try:
                    vidout.write(first_frame)
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        vidout.write(frame)
                finally:
                    vidout.release()
            except Exception:
                if os.path.exists(video_only_path):
                    try:
                        os.remove(video_only_path)
                    except OSError:
                        pass
                raise
        finally:
            cap.release()

        # Mux video + audio from original into final_path
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", video_only_path,
            "-i", temp_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0?",
            "-movflags", "+faststart",
            final_path,
        ]
        try:
            proc = subprocess.run(
                mux_cmd,
                capture_output=True,
                timeout=self.ffmpeg_timeout,
            )
            if proc.returncode != 0:
                logger.warning(f"Mux failed for {event_id}: {proc.stderr.decode()[:500]}")
                if os.path.exists(video_only_path):
                    os.remove(video_only_path)
                return False
            if os.path.exists(video_only_path):
                os.remove(video_only_path)
            os.remove(temp_path)
            logger.info(f"Transcoded clip for {event_id} (NVENC)")
            return True
        except Exception as e:
            logger.warning(f"Mux failed for {event_id}: {e}")
            if os.path.exists(video_only_path):
                try:
                    os.remove(video_only_path)
                except OSError:
                    pass
            return False

    def _transcode_clip_libx264(self, event_id: str, temp_path: str, final_path: str) -> bool:
        """Transcode using ffmpeg libx264 (CPU). Used when GPU path is unavailable."""
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
