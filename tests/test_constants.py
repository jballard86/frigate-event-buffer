"""Tests for shared constants module."""

from frigate_buffer.constants import HTTP_STREAM_CHUNK_SIZE, NON_CAMERA_DIRS


def test_non_camera_dirs_is_frozenset() -> None:
    """NON_CAMERA_DIRS must be a frozenset for immutability and fast membership."""
    assert isinstance(NON_CAMERA_DIRS, frozenset)


def test_non_camera_dirs_contains_expected_names() -> None:
    """NON_CAMERA_DIRS must contain the four known non-camera directory names."""
    expected = {"ultralytics", "yolo_models", "daily_reports", "daily_reviews"}
    assert NON_CAMERA_DIRS == expected


def test_http_stream_chunk_size() -> None:
    """HTTP_STREAM_CHUNK_SIZE is used for streaming responses; default is 8192."""
    assert HTTP_STREAM_CHUNK_SIZE == 8192
