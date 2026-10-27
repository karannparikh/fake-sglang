import pytest

from sglang.srt.server_args import ServerArgs


@pytest.fixture
def server_args():
    def _make(**overrides):
        base = dict(
            model_path="stub-model",
            served_model_name="stub-model",
            # Most unit tests use short prompts and exercise block behavior at
            # token granularity rather than the production-sized default.
            page_size=1,
            fake_num_kv_blocks=8192,
            # Compress the clock hard: these tests assert ordering and ratios,
            # not wall-clock latency, and a real-time sleep model would make
            # the suite take minutes.
            fake_speedup_ratio=200.0,
        )
        base.update(overrides)
        return ServerArgs(**base)

    return _make


@pytest.fixture
def unused_port():
    """A free TCP port. Bound and released, so tests can run concurrently."""
    import socket

    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()
