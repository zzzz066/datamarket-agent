"""Small standard-library backend for the marketplace dashboard."""

from __future__ import annotations

import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

try:
    from .simulation import simulate_dashboard
except ImportError:  # pragma: no cover - supports direct script execution
    from simulation import simulate_dashboard


ROOT = Path(__file__).resolve().parent


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):  # noqa: N802 - stdlib API name
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.path = "/dashboard.html"
            return super().do_GET()
        if parsed.path == "/api/simulate":
            return self._serve_simulation(parsed.query)
        return super().do_GET()

    def _serve_simulation(self, query: str):
        params = parse_qs(query)

        def number(name: str, default: float) -> float:
            try:
                return float(params.get(name, [default])[0])
            except (TypeError, ValueError):
                return default

        payload = simulate_dashboard(
            buyers=int(number("buyers", 120)),
            shade=number("shade", 0.65),
            overbid=number("overbid", 1.4),
            lambda_penalty=number("lambda", 0.7),
        )
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8000), DashboardHandler)
    print("Serving dashboard at http://127.0.0.1:8000/")
    server.serve_forever()


if __name__ == "__main__":
    main()
