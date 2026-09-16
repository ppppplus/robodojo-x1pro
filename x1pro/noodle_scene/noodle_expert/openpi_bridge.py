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


# Exact episode prompts stored with checkpoint 29999. The reference follower
# poses and gripper ranges come from representative episodes in the matching
# filtered real-robot dataset. They anchor live simulator motion in the
# X2Robot controller coordinate system.
TASK_SPECS: dict[str, dict[str, Any]] = {
    "place_noodles_in_pot": {
        "prompt": "Pick up the noodles from the white plate and place them into the pot.",
        "left": [-0.00182585, 0.01061521, 0.00297701, -0.06088022, 0.01425371, 0.24652328, -0.00782013],
        "right": [0.05300714, 0.02185267, 0.11441533, 0.25553318, 0.62425712, 0.54150861, -0.01430511],
        "gripper": {"left": [-0.00782013, 4.50], "right": [-0.01430511, 3.96562958]},
    },
    "sprinkle_chili_seasoning": {
        "prompt": "Pick up the pink chili shaker, sprinkle chili seasoning into the white bowl, then return the shaker to its starting spot.",
        "left": [0.00506686, 0.00346641, 0.00060374, -0.08118435, 0.02541496, -0.13610072, -0.02040863],
        "right": [-0.00157412, -0.00880805, 0.01528519, -0.01007537, -0.08731259, -0.19647355, 0.01544952],
        "gripper": {"left": [-0.02040863, 4.50694275], "right": [0.01544952, 3.97]},
    },
    "sprinkle_green_onions": {
        "prompt": "Pick up the green bottle with the yellow spout, dispense green onion seasoning into the white bowl, then return the bottle to its starting spot.",
        "left": [0.00005827, 0.00348920, 0.00995069, 0.05389323, -0.07388153, 0.09218256, -0.01964569],
        "right": [-0.00126885, 0.01987366, 0.00886175, -0.16789634, -0.06886999, 0.10000016, 0.01544952],
        "gripper": {"left": [-0.05283451, 4.54547119], "right": [0.01544952, 3.97]},
    },
    "sprinkle_salt": {
        "prompt": "Pick up the black salt shaker, sprinkle salt into the white bowl, then return the shaker to its starting spot.",
        "left": [0.00149563, 0.01004708, 0.01184064, -0.19150748, -0.08546896, -0.09508662, -0.02117157],
        "right": [-0.00058851, 0.00134421, 0.00242485, -0.12456199, 0.00015416, -0.03262997, 0.01544952],
        "gripper": {"left": [-0.04062748, 4.49092102], "right": [0.01544952, 3.97]},
    },
    "transfer_noodles_to_bowl": {
        "prompt": "Pick up the metal strainer and transfer the noodles from the pot into the white bowl.",
        "left": [-0.00177138, 0.00167804, 0.00349001, -0.11692190, 0.00681590, 0.16067548, -0.00782013],
        "right": [-0.00877922, 0.01310323, 0.05473353, 0.04043579, -0.35228738, 0.07738906, -0.00705719],
        "gripper": {"left": [-0.00782013, 4.50], "right": [-0.00705719, 3.52540588]},
    },
}


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
    """Match the no-master-arm X1 Pro takeover sequence used on the real robot."""

    history_size = 3
    future_size = 3
    slave_dim = 14
    master_dim = 14
    action_dim = 29
    latency_steps = 3
    move_steps = 10

    def __init__(self, initial_slave: np.ndarray, phase: float = 0.0) -> None:
        slave = self._vector(initial_slave, self.slave_dim, "initial_slave")
        master_phase = np.concatenate((slave, [phase]), dtype=np.float32)
        self._slave_history: deque[np.ndarray] = deque(
            [slave.copy()] * (self.history_size + 1), maxlen=self.history_size + 1
        )
        self._master_queue: deque[np.ndarray] = deque(
            [master_phase.copy()] * (self.history_size + 1 + self.future_size), maxlen=100
        )
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
        history = list(self._slave_history)
        history[-1] = slave.copy()
        slave_rows = [*history[-(self.history_size + 1):], *([slave.copy()] * self.future_size)]
        master_rows = list(self._master_queue)[-(self.history_size + 1 + self.future_size):]
        state = np.stack(
            [
                np.concatenate((s, m[:self.master_dim], [m[self.master_dim]]), dtype=np.float32)
                for s, m in zip(slave_rows, master_rows)
            ]
        ).astype(np.float32)
        if state.shape != (self.history_size + 1 + self.future_size, self.action_dim):
            raise AssertionError(state.shape)
        return {"images": images, "prompt": prompt, "state": state}

    def commit(self, policy_result: dict[str, Any], max_steps: int | None = None) -> np.ndarray:
        actions = np.asarray(policy_result["actions"], dtype=np.float32)
        if actions.ndim != 2 or actions.shape[1] < self.action_dim:
            raise ValueError(f"expected [horizon,{self.action_dim}] actions, got {actions.shape}")
        self.predicted_actions = actions[:, :self.action_dim].copy()
        end = self.latency_steps + self.move_steps
        planned = self.predicted_actions[self.latency_steps:end, self.slave_dim:self.action_dim].copy()
        if max_steps is not None:
            planned = planned[:max_steps]
        if len(planned) == 0:
            raise ValueError(f"action horizon {len(actions)} is too short for latency {self.latency_steps}")
        self._master_queue.extend(planned)
        self.phase = float(planned[-1, self.master_dim])
        return planned

    def record_slave(self, slave: np.ndarray) -> None:
        self._slave_history.append(self._vector(slave, self.slave_dim, "slave").copy())
