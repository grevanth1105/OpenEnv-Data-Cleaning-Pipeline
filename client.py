"""
client.py — HTTP/WebSocket client for Data Cleaning Pipeline OpenEnv
Provides both async and sync interfaces following the OpenEnv pattern.
"""

from __future__ import annotations

import json
import uuid
from contextlib import contextmanager, asynccontextmanager
from typing import Any, Dict, Optional, Tuple

import requests
import websockets

from models import (
    DataCleaningAction,
    DataCleaningObservation,
    DataCleaningState,
    GraderResult,
)


class DataCleaningEnv:
    """
    HTTP client for the Data Cleaning Pipeline environment.

    Usage — Sync:
        with DataCleaningEnv(base_url="http://localhost:8000").sync() as env:
            obs = env.reset(task_name="missing_value_imputation")
            obs, reward, done, info = env.step(
                DataCleaningAction(action_type="impute", column="age",
                                   params={"strategy": "median"})
            )

    Usage — Async:
        async with DataCleaningEnv(base_url="http://localhost:8000") as env:
            obs = await env.reset(task_name="missing_value_imputation")
            obs, reward, done, info = await env.step(...)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        session_id: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        self.base_url   = base_url.rstrip("/")
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.timeout    = timeout
        self._ws        = None          # WebSocket connection (async)
        self._session   = None          # requests.Session (sync)

    # ------------------------------------------------------------------
    # Async interface
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "DataCleaningEnv":
        ws_url = self.base_url.replace("http", "ws") + "/ws"
        self._ws = await websockets.connect(ws_url, ping_interval=20)
        return self

    async def __aexit__(self, *_) -> None:
        if self._ws:
            await self._ws.close()
            self._ws = None

    async def reset(
        self,
        task_name: str = "missing_value_imputation",
        seed: int = 42,
    ) -> DataCleaningObservation:
        if self._ws:
            return await self._ws_reset(task_name, seed)
        return self._http_reset(task_name, seed)

    async def step(
        self,
        action: DataCleaningAction,
    ) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        if self._ws:
            return await self._ws_step(action)
        return self._http_step(action)

    async def state(self) -> DataCleaningState:
        if self._ws:
            return await self._ws_state()
        return self._http_state()

    async def grader(self) -> GraderResult:
        if self._ws:
            return await self._ws_grader()
        return self._http_grader()

    # ------------------------------------------------------------------
    # WebSocket internals
    # ------------------------------------------------------------------

    async def _ws_reset(self, task_name: str, seed: int) -> DataCleaningObservation:
        await self._ws.send(json.dumps({
            "type":      "reset",
            "task_name": task_name,
            "seed":      seed,
        }))
        resp = json.loads(await self._ws.recv())
        return DataCleaningObservation(**resp["observation"])

    async def _ws_step(
        self, action: DataCleaningAction
    ) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        await self._ws.send(json.dumps({
            "type":   "step",
            "action": action.dict(),
        }))
        resp = json.loads(await self._ws.recv())
        return (
            DataCleaningObservation(**resp["observation"]),
            resp["reward"],
            resp["done"],
            resp.get("info", {}),
        )

    async def _ws_state(self) -> DataCleaningState:
        await self._ws.send(json.dumps({"type": "state"}))
        resp = json.loads(await self._ws.recv())
        return DataCleaningState(**resp["state"])

    async def _ws_grader(self) -> GraderResult:
        await self._ws.send(json.dumps({"type": "grader"}))
        resp = json.loads(await self._ws.recv())
        return GraderResult(**resp["result"])

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def sync(self) -> "_SyncDataCleaningEnv":
        return _SyncDataCleaningEnv(self.base_url, self.session_id, self.timeout)

    # ------------------------------------------------------------------
    # HTTP internals (stateless fallback)
    # ------------------------------------------------------------------

    def _http_reset(self, task_name: str, seed: int) -> DataCleaningObservation:
        resp = requests.post(
            f"{self.base_url}/reset",
            json={"task_name": task_name, "seed": seed, "session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return DataCleaningObservation(**resp.json())

    def _http_step(
        self, action: DataCleaningAction
    ) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        resp = requests.post(
            f"{self.base_url}/step",
            json={"action": action.dict(), "session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            DataCleaningObservation(**data["observation"]),
            data["reward"],
            data["done"],
            data.get("info", {}),
        )

    def _http_state(self) -> DataCleaningState:
        resp = requests.get(
            f"{self.base_url}/state",
            params={"session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return DataCleaningState(**resp.json())

    def _http_grader(self) -> GraderResult:
        resp = requests.post(
            f"{self.base_url}/grader",
            json={"session_id": self.session_id},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return GraderResult(**resp.json())

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def tasks(self) -> Dict[str, Any]:
        """Fetch task list and action schema from /tasks."""
        resp = requests.get(f"{self.base_url}/tasks", timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def baseline(self, seed: int = 42) -> Dict[str, Any]:
        """Trigger heuristic baseline and return scores for all 3 tasks."""
        resp = requests.post(
            f"{self.base_url}/baseline",
            json={"seed": seed},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()

    def health(self) -> bool:
        """Return True if server is healthy."""
        try:
            resp = requests.get(f"{self.base_url}/health", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    @classmethod
    async def from_env(cls, repo_id: str, **kwargs) -> "DataCleaningEnv":
        """
        Auto-pull and run environment from HF Space.
        e.g. DataCleaningEnv.from_env("openenv/data-cleaning-env")
        """
        base_url = f"https://{repo_id.replace('/', '-')}.hf.space"
        return cls(base_url=base_url, **kwargs)


# ------------------------------------------------------------------
# Sync wrapper class
# ------------------------------------------------------------------

class _SyncDataCleaningEnv:
    """
    Synchronous context manager wrapper around DataCleaningEnv HTTP calls.

    with DataCleaningEnv("http://localhost:8000").sync() as env:
        obs = env.reset()
        obs, reward, done, info = env.step(action)
    """

    def __init__(self, base_url: str, session_id: str, timeout: int) -> None:
        self._base_url   = base_url.rstrip("/")
        self._session_id = session_id
        self._timeout    = timeout
        self._session    = requests.Session()

    def __enter__(self) -> "_SyncDataCleaningEnv":
        return self

    def __exit__(self, *_) -> None:
        self._session.close()

    def reset(
        self,
        task_name: str = "missing_value_imputation",
        seed: int = 42,
    ) -> DataCleaningObservation:
        resp = self._post("/reset", {
            "task_name":  task_name,
            "seed":       seed,
            "session_id": self._session_id,
        })
        return DataCleaningObservation(**resp)

    def step(
        self,
        action: DataCleaningAction,
    ) -> Tuple[DataCleaningObservation, float, bool, Dict]:
        resp = self._post("/step", {
            "action":     action.dict(),
            "session_id": self._session_id,
        })
        return (
            DataCleaningObservation(**resp["observation"]),
            resp["reward"],
            resp["done"],
            resp.get("info", {}),
        )

    def state(self) -> DataCleaningState:
        resp = self._get("/state", params={"session_id": self._session_id})
        return DataCleaningState(**resp)

    def grader(self) -> GraderResult:
        resp = self._post("/grader", {"session_id": self._session_id})
        return GraderResult(**resp)

    def tasks(self) -> Dict[str, Any]:
        return self._get("/tasks")

    def health(self) -> bool:
        try:
            r = self._session.get(f"{self._base_url}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: Dict) -> Dict:
        r = self._session.post(
            f"{self._base_url}{path}",
            json=payload,
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()

    def _get(self, path: str, params: Optional[Dict] = None) -> Dict:
        r = self._session.get(
            f"{self._base_url}{path}",
            params=params or {},
            timeout=self._timeout,
        )
        r.raise_for_status()
        return r.json()


# ------------------------------------------------------------------
# Smoke test (requires running server)
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    print(f"Testing client against: {base_url}\n")

    with DataCleaningEnv(base_url).sync() as env:
        # Health check
        ok = env.health()
        print(f"  Health   : {'✅ OK' if ok else '❌ FAIL'}")
        if not ok:
            print("  Server not reachable. Start with: uvicorn server.app:app --port 8000")
            sys.exit(1)

        # Tasks
        tasks = env.tasks()
        print(f"  Tasks    : {[t['task_name'] for t in tasks['tasks']]}")

        # Episode
        obs = env.reset(task_name="missing_value_imputation", seed=42)
        print(f"  Reset    : rows={obs.total_rows} issues={obs.issues_remaining}")

        action = DataCleaningAction(
            action_type="impute", column="age", params={"strategy": "median"}
        )
        obs, reward, done, info = env.step(action)
        print(f"  Step     : reward={reward} done={done} progress={obs.progress_pct:.0%}")

        result = env.grader()
        print(f"  Grader   : score={result.score:.4f} passed={result.passed}")

    print("\n✅ Client working correctly!")
