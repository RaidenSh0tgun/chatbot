import os
import queue
import signal
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext


APP_DIR = os.path.dirname(os.path.abspath(__file__))
CHATBOT_CMD = ["python3", "-u", "main.py"]
TUNNEL_CMD = ["cloudflared", "tunnel", "run", "spaabot"]


class ChatbotControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SPAA Chatbot Control")
        self.root.geometry("860x560")
        self.root.minsize(720, 460)

        self.chatbot_process = None
        self.tunnel_process = None
        self.log_queue = queue.Queue()

        self._build_ui()
        self._poll_logs()
        self._refresh_status()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self):
        self.root.configure(bg="#f6f7f9")

        header = tk.Frame(self.root, bg="#f6f7f9")
        header.pack(fill="x", padx=24, pady=(22, 12))

        title = tk.Label(
            header,
            text="SPAA Chatbot Control",
            font=("Helvetica", 22, "bold"),
            bg="#f6f7f9",
            fg="#18202a",
        )
        title.pack(side="left")

        status_bar = tk.Frame(header, bg="#f6f7f9")
        status_bar.pack(side="right")

        self.chatbot_dot = tk.Canvas(status_bar, width=18, height=18, bg="#f6f7f9", highlightthickness=0)
        self.chatbot_dot.grid(row=0, column=0, padx=(0, 6))
        tk.Label(status_bar, text="Chatbot", font=("Helvetica", 12), bg="#f6f7f9", fg="#18202a").grid(row=0, column=1, padx=(0, 18))

        self.tunnel_dot = tk.Canvas(status_bar, width=18, height=18, bg="#f6f7f9", highlightthickness=0)
        self.tunnel_dot.grid(row=0, column=2, padx=(0, 6))
        tk.Label(status_bar, text="Tunnel", font=("Helvetica", 12), bg="#f6f7f9", fg="#18202a").grid(row=0, column=3)

        controls = tk.Frame(self.root, bg="#f6f7f9")
        controls.pack(fill="x", padx=24, pady=(0, 16))

        self.toggle_button = tk.Button(
            controls,
            text="Turn on Chatbot",
            command=self._toggle_chatbot,
            font=("Helvetica", 15, "bold"),
            bg="#2364aa",
            fg="white",
            activebackground="#1d4f85",
            activeforeground="white",
            relief="flat",
            padx=22,
            pady=12,
            cursor="hand2",
        )
        self.toggle_button.pack(side="left")

        self.state_text = tk.Label(
            controls,
            text="Stopped",
            font=("Helvetica", 12),
            bg="#f6f7f9",
            fg="#58606d",
        )
        self.state_text.pack(side="left", padx=14)

        log_frame = tk.Frame(self.root, bg="#d9dde4", bd=1, relief="solid")
        log_frame.pack(fill="both", expand=True, padx=24, pady=(0, 22))

        log_header = tk.Frame(log_frame, bg="#eef1f5")
        log_header.pack(fill="x")
        tk.Label(
            log_header,
            text="main.py terminal output",
            font=("Helvetica", 12, "bold"),
            bg="#eef1f5",
            fg="#18202a",
            padx=12,
            pady=8,
        ).pack(side="left")

        clear_button = tk.Button(
            log_header,
            text="Clear",
            command=self._clear_logs,
            font=("Helvetica", 11),
            bg="#eef1f5",
            fg="#2364aa",
            activebackground="#dfe4eb",
            activeforeground="#1d4f85",
            relief="flat",
            padx=10,
            pady=4,
            cursor="hand2",
        )
        clear_button.pack(side="right", padx=8)

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            wrap="word",
            font=("Menlo", 11),
            bg="#101820",
            fg="#edf2f7",
            insertbackground="#edf2f7",
            relief="flat",
            padx=12,
            pady=12,
        )
        self.log_text.pack(fill="both", expand=True)
        self.log_text.configure(state="disabled")

    def _toggle_chatbot(self):
        if self._is_running(self.chatbot_process) or self._is_running(self.tunnel_process):
            self._stop_processes()
        else:
            self._start_processes()
        self._refresh_status()

    def _start_processes(self):
        self._append_log_line("$ python3 main.py\n")
        if not os.path.exists(os.path.join(APP_DIR, "main.py")):
            self._append_log_line("main.py was not found in this folder. Save or add main.py, then try again.\n")
            self._refresh_status()
            return

        try:
            self.chatbot_process = subprocess.Popen(
                CHATBOT_CMD,
                cwd=APP_DIR,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )
            threading.Thread(target=self._read_chatbot_output, daemon=True).start()
        except Exception as exc:
            self._append_log_line("Could not start chatbot: {0}\n".format(exc))
            self.chatbot_process = None
            return

        try:
            self.tunnel_process = subprocess.Popen(
                TUNNEL_CMD,
                cwd=APP_DIR,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )
        except Exception as exc:
            self._append_log_line("Could not start Cloudflare tunnel: {0}\n".format(exc))
            self.tunnel_process = None

    def _stop_processes(self):
        self._append_log_line("\nStopping chatbot and tunnel...\n")
        self._terminate_process(self.chatbot_process)
        self._terminate_process(self.tunnel_process)
        self.chatbot_process = None
        self.tunnel_process = None

    def _terminate_process(self, process):
        if not self._is_running(process):
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

    def _read_chatbot_output(self):
        if not self.chatbot_process or not self.chatbot_process.stdout:
            return

        for line in self.chatbot_process.stdout:
            self.log_queue.put(line)

        self.log_queue.put("\nmain.py stopped.\n")

    def _poll_logs(self):
        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            self._append_log_line(line)

        self.root.after(100, self._poll_logs)

    def _append_log_line(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line)
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_logs(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _refresh_status(self):
        chatbot_running = self._is_running(self.chatbot_process)
        tunnel_running = self._is_running(self.tunnel_process)

        self._draw_dot(self.chatbot_dot, chatbot_running)
        self._draw_dot(self.tunnel_dot, tunnel_running)

        if chatbot_running or tunnel_running:
            self.toggle_button.configure(text="Turn off Chatbot", bg="#b12a34", activebackground="#8e2028")
            if chatbot_running and tunnel_running:
                self.state_text.configure(text="Chatbot and tunnel are running", fg="#1f7a3d")
            elif chatbot_running:
                self.state_text.configure(text="Chatbot is running; tunnel is off", fg="#a16600")
            else:
                self.state_text.configure(text="Tunnel is running; chatbot is off", fg="#a16600")
        else:
            self.toggle_button.configure(text="Turn on Chatbot", bg="#2364aa", activebackground="#1d4f85")
            self.state_text.configure(text="Stopped", fg="#58606d")

        self.root.after(1000, self._refresh_status)

    def _draw_dot(self, canvas, is_on):
        canvas.delete("all")
        fill = "#22a04a" if is_on else "#c7313b"
        canvas.create_oval(2, 2, 16, 16, fill=fill, outline="")

    def _is_running(self, process):
        return process is not None and process.poll() is None

    def _on_close(self):
        if self._is_running(self.chatbot_process) or self._is_running(self.tunnel_process):
            should_quit = messagebox.askyesno(
                "Quit SPAA Chatbot Control",
                "The chatbot or tunnel is still running. Stop them and quit?",
            )
            if not should_quit:
                return
            self._stop_processes()

        self.root.destroy()


def main():
    root = tk.Tk()
    ChatbotControlApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
