import numpy as np

from typing import Union, Protocol, TypedDict, Any, Tuple


class _margs_t(TypedDict):
    model_type: str
    in_channels: int
    drop_out: int
    learning_rate: float
    weight_decay: float
    device: float
    use_attention: bool


class _targs_t(TypedDict):
    n_iterations: int
    buffer_size: int
    n_playout: int
    board_size: int
    update_times_per_episode: int
    log_freq: int
    batch_size: int


class _env_t(Protocol):
    board_size: int
    availables: list
    def reset(self, seed: int) -> None: ...
    def current_state(self) -> np.ndarray: ... 
    def step(self, idx: int) -> None: ... 
    def render(self, render: str) -> None: ...
    def terminated(self) -> Tuple[bool, int]: ...


class _agent_t(Protocol):
    def get_action(self, env) -> Any: ...
    def reset(self) -> Any: ...
    