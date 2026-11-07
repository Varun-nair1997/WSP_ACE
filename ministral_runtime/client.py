"""Start, initialize, and query a local or remote Ollama-served Ministral model."""

from __future__ import annotations

import html
import itertools
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass
class OllamaConfig:
    """Runtime configuration. Set OLLAMA_URL when connecting to a remote server."""
    model: str = "ministral-3:8b"
    base_url: str | None = None
    ollama_bin: str | None = None
    host: str = "0.0.0.0"
    port: int = 11434
    keep_alive: str = "30m"

    def resolved_url(self):
        if self.base_url:
            return self.base_url.rstrip("/")
        hostname = subprocess.check_output(["hostname", "-f"], text=True).strip()
        return f"http://{hostname}:{self.port}"

    def resolved_binary(self):
        candidate = self.ollama_bin or os.environ.get("OLLAMA_BIN")
        if candidate:
            return Path(candidate).expanduser()
        found = shutil.which("ollama")
        return Path(found) if found else None


class OllamaRuntime:
    """Manage an Ollama process and issue generate requests."""

    def __init__(self, config: OllamaConfig | None = None):
        self.config = config or OllamaConfig()
        self.base_url = self.config.resolved_url()
        self.process = None
        self._log_handle = None

    def start_server(self, log_path="ollama.log", bind_host=None):
        """Start Ollama if requested. On managed clusters, a prestarted service may be preferable."""
        binary = self.config.resolved_binary()
        if binary is None or not binary.is_file():
            raise FileNotFoundError(
                "Ollama executable not found. Set OllAMA_BIN/OLLAMA_BIN or provide ollama_bin."
            )
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"{bind_host or self.config.host}:{self.config.port}"
        log_file = Path(log_path).expanduser()
        log_file.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = open(log_file, "a", encoding="utf-8")
        self.process = subprocess.Popen(
            [str(binary), "serve"], env=env,
            stdout=self._log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        return self.process.pid

    def wait_until_ready(self, attempts=30, interval=2, timeout=5):
        """Poll the Ollama tags endpoint until it responds."""
        last_error = None
        for _ in range(attempts):
            try:
                response = requests.get(f"{self.base_url}/api/tags", timeout=timeout)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as exc:
                last_error = exc
                time.sleep(interval)
        raise RuntimeError(f"Ollama did not become ready at {self.base_url}") from last_error

    def load_model(self, timeout=(30, 600)):
        """Warm the selected model with a minimal generation request."""
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.config.model,
                "prompt": "",
                "stream": False,
                "keep_alive": self.config.keep_alive,
                "options": {"num_predict": 1},
            },
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _format_time(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m {seconds:02d}s"
        if minutes:
            return f"{minutes}m {seconds:02d}s"
        return f"{seconds}s"

    def generate(self, prompt, options=None, timeout=3600, stream=False, progress=True):
        """Generate a response; optionally render a notebook progress panel and metrics."""
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": stream,
            "keep_alive": self.config.keep_alive,
        }
        if options is not None:
            payload["options"] = options

        if not progress:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(30, timeout),
            )
            response.raise_for_status()
            return response.json()

        try:
            from IPython.display import HTML, display
        except ImportError as exc:
            raise RuntimeError("Notebook progress requires IPython; use progress=False otherwise.") from exc

        start = time.monotonic()
        stop_event = threading.Event()
        frames = itertools.cycle(["◐", "◓", "◑", "◒"])
        handle = display(HTML("<div>Initializing inference...</div>"), display_id=True)

        def update():
            while not stop_event.is_set():
                elapsed = time.monotonic() - start
                remaining = max(timeout - elapsed, 0)
                spinner = next(frames)
                color = "#ef4444" if remaining <= 60 else "#d97706" if remaining <= timeout * .1 else "#3b82f6"
                status = "TIMEOUT APPROACHING" if remaining <= 60 else "TIMEOUT WARNING" if remaining <= timeout * .1 else "INFERENCE RUNNING"
                panel = f"""
                <div style="padding:14px;border-radius:9px;background:#211432;color:#fff;border:1px solid #493568;font-family:sans-serif">
                  <b style="color:{color}">{spinner} {status}</b>
                  <div style="margin-top:8px;color:#d1d1d8">Elapsed: {self._format_time(elapsed)}</div>
                  <div style="color:#d1d1d8">Approx. read-timeout budget remaining: {self._format_time(remaining)}</div>
                </div>"""
                handle.update(HTML(panel))
                stop_event.wait(.5)

        thread = threading.Thread(target=update, daemon=True)
        thread.start()
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=(30, timeout),
            )
            response.raise_for_status()
            result = response.json()
        except Exception as exc:
            handle.update(HTML(
                '<div style="color:#ef4444;font-weight:700">INFERENCE FAILED</div>'
                f'<pre>{html.escape(type(exc).__name__ + ": " + str(exc))}</pre>'
            ))
            raise
        finally:
            stop_event.set()
            thread.join(timeout=2)

        result["_python_wall_time_seconds"] = time.monotonic() - start
        self._display_metrics(result)
        return result

    def _display_metrics(self, result):
        try:
            from IPython.display import HTML, display
        except ImportError:
            return
        def seconds(key):
            value = result.get(key)
            return "N/A" if value is None else f"{value / 1e9:.2f} s"
        def rate(tokens_key, duration_key):
            tokens, duration = result.get(tokens_key), result.get(duration_key)
            if tokens is None or not duration:
                return "N/A"
            return f"{tokens / (duration / 1e9):.2f} tokens/s"
        rows = [
            ("Model", result.get("model", self.config.model)),
            ("Python wall time", f"{result.get('_python_wall_time_seconds', 0):.2f} s"),
            ("Total inference", seconds("total_duration")),
            ("Model loading", seconds("load_duration")),
            ("Prompt processing", seconds("prompt_eval_duration")),
            ("Prompt rate", rate("prompt_eval_count", "prompt_eval_duration")),
            ("Text generation", seconds("eval_duration")),
            ("Generation rate", rate("eval_count", "eval_duration")),
            ("Input tokens", result.get("prompt_eval_count", "N/A")),
            ("Output tokens", result.get("eval_count", "N/A")),
            ("Completion reason", result.get("done_reason", "Not reported")),
        ]
        body = "".join(
            f"<tr><td style='padding:6px;color:#aaa'>{html.escape(str(k))}</td>"
            f"<td style='padding:6px;color:white;text-align:right'>{html.escape(str(v))}</td></tr>"
            for k, v in rows
        )
        display(HTML(
            "<div style='padding:14px;background:#211432;border:1px solid #493568;"
            "border-radius:9px;color:white;font-family:sans-serif'>"
            "<b>Post-inference performance report</b>"
            f"<table style='width:100%;margin-top:8px'>{body}</table></div>"
        ))

    def close_log(self):
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None
