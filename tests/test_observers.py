import numpy as np

from environment.observers import TrafficObserver


def test_normalize_state_shape():
    observer = TrafficObserver(max_lanes=12)
    state = observer.normalize_state(
        queues=[5, 3, 2],
        wait_times=[10.0, 20.0, 5.0],
        current_phase=2,
        phase_duration=15,
    )
    assert state.shape == (26,)
    assert state.dtype == np.float32


def test_normalize_with_speeds():
    observer = TrafficObserver(max_lanes=12)
    state = observer.normalize_state(
        queues=[1, 2],
        wait_times=[5.0, 10.0],
        current_phase=0,
        phase_duration=5,
        speeds=[8.0, 12.0],
    )
    assert state.shape == (38,)


def test_build_global_state():
    observer = TrafficObserver()
    local = {"A": np.zeros(26), "B": np.ones(26) * 0.5}
    global_state = observer.build_global_state(local)
    assert global_state.shape == (52,)
