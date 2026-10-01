"""Private local engine started and stopped by the Windows desktop shell."""
import argparse
import hmac
import json
import os
from pathlib import Path
import socket
import threading
import time

import uvicorn
from fastapi import Request
from fastapi.responses import JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from app.main import create_app
from app.core.config import Settings


def create_desktop_app(config, frontend_dir: Path, token: str, origin: str):
    if len(token) < 32:
        raise RuntimeError("The desktop shell must supply a private session token.")
    application = create_app(config)

    @application.middleware("http")
    async def desktop_session(request: Request, call_next):
        supplied = request.cookies.get("edgelens_session") or request.headers.get("x-edgelens-session", "")
        if not hmac.compare_digest(supplied, token):
            return JSONResponse({"detail": "This engine belongs to the EdgeLens desktop application."}, status_code=403)
        if request.method not in {"GET", "HEAD", "OPTIONS"} and request.headers.get("origin", origin) != origin:
            return JSONResponse({"detail": "Untrusted request origin."}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        return response

    @application.get("/config.js", include_in_schema=False)
    def desktop_config():
        return Response("window.EDGELENS_CONFIG = Object.freeze({apiBaseUrl:'/api/v1',desktop:true});", media_type="application/javascript")

    application.mount("/", StaticFiles(directory=Path(frontend_dir).resolve(), html=True), name="desktop-ui")
    return application


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frontend-dir", required=True)
    arguments = parser.parse_args()
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(128)
    port = listener.getsockname()[1]
    application = create_desktop_app(Settings.from_environment(), Path(arguments.frontend_dir), os.environ.get("EDGELENS_DESKTOP_TOKEN", ""), f"http://127.0.0.1:{port}")
    server = uvicorn.Server(uvicorn.Config(application, log_level="warning", access_log=False))
    def announce_ready():
        while not server.started and not server.should_exit:
            time.sleep(.05)
        if server.started:
            print(json.dumps({"edgelens_port": port}), flush=True)
    threading.Thread(target=announce_ready, daemon=True).start()
    server.run(sockets=[listener])


if __name__ == "__main__":
    main()
