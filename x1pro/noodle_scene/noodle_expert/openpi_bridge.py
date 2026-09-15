"""OpenPI WebSocket bridge for the X1 Pro ``smp2smp`` noodle checkpoint.

The policy was trained on 15 Hz X2Robot samples.  Each observation contains
three RGB cameras and seven state rows: three historic rows, the present row,
and three future slots.  Future slave state is deliberately copied from the
present state; future master state is filled by the preceding action chunk.
"""

from __future__ import annotations

from collections import deque
import functools
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client


def _pack_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "VOc":
            raise ValueError(f"unsupported array dtype: {value.dtype}")
        return {b"__ndarray__": True, b"data": value.tobytes(), b"dtype": value.dtype.str, b"shape": value.shape}
    if isinstance(value, np.generic):
        return {b"__npgeneric__": True, b"data": value.item(), b"dtype": value.dtype.str}
    return value


def _unpack_array(value: dict[bytes, Any]) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(buffer=value[b"data"], dtype=np.dtype(value[b"dtype"]), shape=value[b"shape"])
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_pack = functools.partial(msgpack.packb, default=_pack_array)
_unpack = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class OpenPiClient:
    """Minimal dependency-free client for OpenPI's MsgPack WebSocket service."""

    def __init__(self, host: str = "127.0.0.1", port: int = 8000) -> None:
        self._ws = websockets.sync.client.connect(f"ws://{host}:{port}", compression=None, max_size=None)
        self.metadata = _unpack(self._ws.recv())

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self._ws.send(_pack(observation))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI inference error:\n{response}")
        return _unpack(response)

    def close(self) -> None:
        self._ws.close()


class Smp2SmpSequence:
    """State sequencing for the 14D slave + 14D master + phase policy."""

    history_size = 3
    future_size = 3
    slave_dim = 14
    master_dim = 14
    action_dim = 29

    def __init__(self, initial_slave: np.ndarray, initial_master: np.ndarray | None = None, phase: float = 0.0) -> None:
        slave = self._vector(initial_slave, self.slave_dim, "initial_slave")
        master = slave.copy() if initial_master is None else self._vector(initial_master, self.master_dim, "initial_master")
        self._past: deque[tuple[np.ndarray, np.ndarray]] = deque(
            [(slave.copy(), master.copy())] * (self.history_size + 1), maxlen=self.history_size + 1
        )
        self.master = master
        self.phase = float(phase)
        self.predicted_actions: np.ndarray | None = None

    @staticmethod
    def _vector(value: np.ndarray, dim: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float32).reshape(-1)
        if vector.size != dim:
            raise ValueError(f"{name} must have {dim} values, got {vector.size}")
        return vector

    def observation(self, slave: np.ndarray, images: dict[str, np.ndarray], prompt: str) -> dict[str, Any]:
        slave = self._vector(slave, self.slave_dim, "slave")
        current = np.concatenate((slave, self.master, [self.phase]), dtype=np.float32)
        historical = [np.concatenate((old_slave, old_master, [self.phase]), dtype=np.float32) for old_slave, old_master in self._past]
        future_master = self._future_master()
        future = [np.concatenate((slave, master, [self.phase]), dtype=np.float32) for master in future_master]
        state = np.stack([*historical[-self.history_size:], current, *future]).astype(np.float32)
        if state.shape != (self.history_size + 1 + self.future_size, self.action_dim):
            raise AssertionError(state.shape)
        return {"images": images, "prompt": prompt, "state": state}

    def commit(self, slave: np.ndarray, policy_result: dict[str, Any]) -> np.ndarray:
        actions = np.asarray(policy_result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise ValueError(f"expected [horizon,{self.action_dim}] actions, got {actions.shape}")
        self.predicted_actions = actions[:, :self.action_dim].copy()
        self.master = self.predicted_actions[0, self.slave_dim : self.slave_dim + self.master_dim].copy()
        self.phase = float(self.predicted_actions[0, 28])
        self._past.append((self._vector(slave, self.slave_dim, "slave").copy(), self.master.copy()))
        return self.master.copy()

    def _future_master(self) -> list[np.ndarray]:
        if self.predicted_actions is None:
            return [self.master.copy() for _ in range(self.future_size)]
        masters = self.predicted_actions[:, self.slave_dim : self.slave_dim + self.master_dim]
        return [masters[min(index + 1, len(masters) - 1)].copy() for index in range(self.future_size)]
