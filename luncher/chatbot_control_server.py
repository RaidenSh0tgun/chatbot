import atexit
import json
import os
import signal
import socket
import subprocess
import threading
import time
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse


APP_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_DIR = os.path.join(APP_DIR, ".venv")
VENV_BIN_DIR = os.path.join(VENV_DIR, "bin")
VENV_PYTHON = os.path.join(VENV_BIN_DIR, "python")
CHATBOT_CMD = [VENV_PYTHON, "-u", "main.py"] if os.path.exists(VENV_PYTHON) else ["python3", "-u", "main.py"]
TUNNEL_CMD = ["cloudflared", "tunnel", "run", "spaabot"]
HOST = "127.0.0.1"
START_PORT = 8765


class ProcessController:
    def __init__(self):
        self.chatbot_process = None
        self.tunnel_process = None
        self.logs = deque(maxlen=2000)
        self.lock = threading.Lock()
        self.server = None

    def start(self):
        with self.lock:
            if self.is_chatbot_running() or self.is_tunnel_running():
                return

            env = self._process_env()
            self._add_log("$ {0} -u main.py".format(CHATBOT_CMD[0]))
            if not os.path.exists(os.path.join(APP_DIR, "main.py")):
                self._add_log("main.py was not found in this folder. Save or add main.py, then try again.")
                return

            try:
                self.chatbot_process = subprocess.Popen(
                    CHATBOT_CMD,
                    cwd=APP_DIR,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    env=env,
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )
                threading.Thread(target=self._read_chatbot_output, daemon=True).start()
            except Exception as exc:
                self._add_log("Could not start chatbot: {0}".format(exc))
                self.chatbot_process = None
                return

            try:
                self.tunnel_process = subprocess.Popen(
                    TUNNEL_CMD,
                    cwd=APP_DIR,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    env=env,
                    preexec_fn=os.setsid if os.name != "nt" else None,
                )
            except Exception as exc:
                self._add_log("Could not start Cloudflare tunnel: {0}".format(exc))
                self.tunnel_process = None

    def stop(self):
        with self.lock:
            self._add_log("")
            self._add_log("Stopping chatbot and tunnel...")
            self._terminate_process(self.chatbot_process)
            self._terminate_process(self.tunnel_process)
            self.chatbot_process = None
            self.tunnel_process = None

    def clear(self):
        with self.lock:
            self.logs.clear()

    def status(self):
        with self.lock:
            chatbot_on = self.is_chatbot_running()
            tunnel_on = self.is_tunnel_running()
            return {
                "chatbotOn": chatbot_on,
                "tunnelOn": tunnel_on,
                "running": chatbot_on or tunnel_on,
                "logs": list(self.logs),
            }

    def shutdown(self):
        self.stop()
        if self.server is not None:
            threading.Thread(target=self.server.shutdown, daemon=True).start()

    def is_chatbot_running(self):
        return self.chatbot_process is not None and self.chatbot_process.poll() is None

    def is_tunnel_running(self):
        return self.tunnel_process is not None and self.tunnel_process.poll() is None

    def _read_chatbot_output(self):
        process = self.chatbot_process
        if not process or not process.stdout:
            return

        for line in process.stdout:
            self._add_log(line.rstrip("\n"))

        self._add_log("")
        self._add_log("main.py stopped.")

    def _add_log(self, line):
        timestamp = time.strftime("%H:%M:%S")
        self.logs.append("[{0}] {1}".format(timestamp, line))

    def _process_env(self):
        env = os.environ.copy()
        if os.path.isdir(VENV_BIN_DIR):
            env["VIRTUAL_ENV"] = VENV_DIR
            env["PATH"] = VENV_BIN_DIR + os.pathsep + env.get("PATH", "")
        return env

    def _terminate_process(self, process):
        if process is None or process.poll() is not None:
            return

        try:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            if os.name != "nt":
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            else:
                process.kill()
        except Exception:
            pass


CONTROLLER = ProcessController()


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SPAA Chatbot Control</title>
  <style>
    :root {
      color-scheme: light;
      --ink: #17202a;
      --muted: #596270;
      --line: #d9dee7;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --blue: #2364aa;
      --blue-dark: #1d4f85;
      --red: #b12a34;
      --red-dark: #8e2028;
      --green: #22a04a;
      --off: #c7313b;
      --term: #101820;
      --term-text: #edf2f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1040px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0;
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 20px;
      margin-bottom: 18px;
    }
    h1 {
      margin: 0;
      font-size: 28px;
      line-height: 1.15;
      letter-spacing: 0;
    }
    .status {
      display: flex;
      align-items: center;
      gap: 18px;
      color: var(--muted);
      font-size: 14px;
      white-space: nowrap;
    }
    .status-item {
      display: inline-flex;
      align-items: center;
      gap: 7px;
    }
    .dot {
      width: 14px;
      height: 14px;
      border-radius: 50%;
      background: var(--off);
      box-shadow: inset 0 0 0 1px rgba(0,0,0,.08);
    }
    .dot.on { background: var(--green); }
    .controls {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 18px;
    }
    button {
      appearance: none;
      border: 0;
      border-radius: 8px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    #toggle {
      min-width: 190px;
      padding: 14px 20px;
      background: var(--blue);
      color: white;
      font-size: 16px;
    }
    #toggle:hover { background: var(--blue-dark); }
    #toggle.running { background: var(--red); }
    #toggle.running:hover { background: var(--red-dark); }
    #clear {
      padding: 9px 12px;
      background: transparent;
      color: var(--blue);
      font-size: 14px;
    }
    #quit {
      padding: 9px 12px;
      background: transparent;
      color: var(--red);
      font-size: 14px;
    }
    #state {
      color: var(--muted);
      font-size: 14px;
    }
    .terminal {
      border: 1px solid var(--line);
      background: var(--panel);
      overflow: hidden;
    }
    .terminal-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      border-bottom: 1px solid var(--line);
      background: #eef1f5;
      padding: 9px 12px;
      font-size: 14px;
      font-weight: 700;
    }
    pre {
      margin: 0;
      height: min(68vh, 620px);
      min-height: 340px;
      overflow: auto;
      padding: 14px;
      background: var(--term);
      color: var(--term-text);
      font: 13px/1.5 Menlo, Monaco, Consolas, "Liberation Mono", monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 680px) {
      header {
        align-items: flex-start;
        flex-direction: column;
      }
      .status {
        width: 100%;
        justify-content: flex-start;
      }
      .controls {
        align-items: stretch;
        flex-direction: column;
      }
      #toggle { width: 100%; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>SPAA Chatbot Control</h1>
      <div class="status">
        <span class="status-item"><span id="chatbot-dot" class="dot"></span>Chatbot</span>
        <span class="status-item"><span id="tunnel-dot" class="dot"></span>Tunnel</span>
      </div>
    </header>
    <section class="controls">
      <button id="toggle" type="button">Turn on Chatbot</button>
      <span id="state">Stopped</span>
    </section>
    <section class="terminal">
      <div class="terminal-head">
        <span>main.py terminal output</span>
        <span>
          <button id="clear" type="button">Clear</button>
          <button id="quit" type="button">Quit App</button>
        </span>
      </div>
      <pre id="logs"></pre>
    </section>
  </main>
  <script>
    const toggle = document.getElementById("toggle");
    const clear = document.getElementById("clear");
    const quit = document.getElementById("quit");
    const state = document.getElementById("state");
    const logs = document.getElementById("logs");
    const chatbotDot = document.getElementById("chatbot-dot");
    const tunnelDot = document.getElementById("tunnel-dot");

    async function api(path, options = {}) {
      const response = await fetch(path, options);
      if (!response.ok) throw new Error(await response.text());
      return response.json();
    }

    function render(data) {
      chatbotDot.classList.toggle("on", data.chatbotOn);
      tunnelDot.classList.toggle("on", data.tunnelOn);
      toggle.classList.toggle("running", data.running);
      toggle.textContent = data.running ? "Turn off Chatbot" : "Turn on Chatbot";

      if (data.chatbotOn && data.tunnelOn) {
        state.textContent = "Chatbot and tunnel are running";
        state.style.color = "#1f7a3d";
      } else if (data.chatbotOn) {
        state.textContent = "Chatbot is running; tunnel is off";
        state.style.color = "#a16600";
      } else if (data.tunnelOn) {
        state.textContent = "Tunnel is running; chatbot is off";
        state.style.color = "#a16600";
      } else {
        state.textContent = "Stopped";
        state.style.color = "#596270";
      }

      const atBottom = logs.scrollHeight - logs.clientHeight - logs.scrollTop < 20;
      logs.textContent = data.logs.join("\\n");
      if (atBottom) logs.scrollTop = logs.scrollHeight;
    }

    async function refresh() {
      try {
        render(await api("/api/status"));
      } catch (error) {
        state.textContent = "Control server is unavailable";
        state.style.color = "#b12a34";
      }
    }

    toggle.addEventListener("click", async () => {
      const endpoint = toggle.classList.contains("running") ? "/api/stop" : "/api/start";
      render(await api(endpoint, { method: "POST" }));
    });

    clear.addEventListener("click", async () => {
      render(await api("/api/clear", { method: "POST" }));
    });

    quit.addEventListener("click", async () => {
      await api("/api/quit", { method: "POST" });
      document.body.innerHTML = "<main><h1>SPAA Chatbot Control</h1><p>The control app is closed.</p></main>";
    });

    refresh();
    setInterval(refresh, 1000);
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        return

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send_html(HTML)
        elif path == "/api/status":
            self._send_json(CONTROLLER.status())
        else:
            self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/start":
            CONTROLLER.start()
            self._send_json(CONTROLLER.status())
        elif path == "/api/stop":
            CONTROLLER.stop()
            self._send_json(CONTROLLER.status())
        elif path == "/api/clear":
            CONTROLLER.clear()
            self._send_json(CONTROLLER.status())
        elif path == "/api/quit":
            self._send_json({"ok": True})
            CONTROLLER.shutdown()
        else:
            self.send_error(404)

    def _send_html(self, html):
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data):
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def find_port(start_port):
    for port in range(start_port, start_port + 50):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((HOST, port))
            except OSError:
                continue
            return port
    raise RuntimeError("No available local port found.")


def main():
    port = find_port(START_PORT)
    server = ThreadingHTTPServer((HOST, port), Handler)
    CONTROLLER.server = server
    atexit.register(CONTROLLER.shutdown)
    url = "http://{0}:{1}".format(HOST, port)
    print("SPAA Chatbot Control is running at {0}".format(url), flush=True)
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    server.serve_forever()


if __name__ == "__main__":
    main()
