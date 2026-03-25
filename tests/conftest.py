"""
Pytest conftest. Optional mocks for tests that run without GPU or optional deps.
GPU decode is mocked per-test (e.g. nvidia.decoder._create_simple_decoder,
VideoService._decoder_context, or get_gpu_backend return values).
"""
