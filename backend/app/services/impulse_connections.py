"""Project-key connections live only in the local engine's memory."""
import secrets
import time
from threading import RLock
from app.services import edge_impulse


class ImpulseConnections:
    def __init__(self, *, clock=time.monotonic):
        self._clock = clock
        self._lock = RLock()
        self._items = {}

    def _expire(self):
        now = self._clock()
        self._items = {k: v for k, v in self._items.items() if v["expires"] > now}

    @staticmethod
    def _public(identifier, item):
        return {"id": identifier, "access": "project_key", "projects": item["projects"]}

    def connect(self, key):
        # Validate against Studio before showing Connected. A project key cannot
        # enumerate all projects in the user's account.
        projects = edge_impulse.list_projects(key)
        if not projects:
            raise ValueError("This key returned no accessible projects. Check its Read + Write permission.")
        with self._lock:
            self._expire()
            if len(self._items) >= 8:
                raise ValueError("Disconnect a project before adding another (eight connections maximum).")
            identifier = "eic_" + secrets.token_urlsafe(24)
            item = {"key": key, "projects": projects, "expires": self._clock() + 8 * 3600}
            self._items[identifier] = item
            return self._public(identifier, item)

    def list(self):
        with self._lock:
            self._expire()
            return [self._public(identifier, item) for identifier, item in self._items.items()]

    def credential(self, identifier, project_id):
        with self._lock:
            self._expire()
            item = self._items.get(identifier)
            if item is None:
                raise ValueError("Edge Impulse connection expired or was disconnected. Connect your project again.")
            project = next((p for p in item["projects"] if p["id"] == project_id), None)
            if project is None:
                raise ValueError("This project is not accessible through the selected connection.")
            return item["key"], project["name"]

    def disconnect(self, identifier):
        with self._lock:
            self._items.pop(identifier, None)

    def close(self):
        with self._lock:
            self._items.clear()
