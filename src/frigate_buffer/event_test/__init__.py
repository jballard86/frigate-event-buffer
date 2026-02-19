"""
Event test: mini orchestrator for running the post-download pipeline (frame extraction, build AI request)
without sending to the proxy. Used by the TEST button on the player for consolidated events.

All test-specific code lives under this package for easy tracking.
"""

from frigate_buffer.event_test.event_test_orchestrator import run_test_pipeline

__all__ = ["run_test_pipeline"]
