#!/usr/bin/env python3
"""
collect_training_data_gui.py

GUI tool for collecting rover training data with a timeline of all saved points.

Saved columns:
- distance_cm
- turn_hint
- label (0=LEFT, 1=RIGHT)

Features:
- Connect to Arduino serial stream (expects DIST:<value>)
- Show latest distance live
- Save labeled samples with LEFT/RIGHT buttons
- Persist to CSV
- Render timeline for ALL points in CSV (distance, hint, label over sample index)

Run:
  python collect_training_data_gui.py
"""

from __future__ import annotations

import csv
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import serial


@dataclass
class Sample:
    distance_cm: int
    turn_hint: int
    label: int


class DistanceReader:
    def __init__(self) -> None:
        self.ser: Optional[serial.Serial] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.latest_distance: Optional[float] = None
        self.last_update_ts: Optional[float] = None

    def connect(self, port: str, baud: int) -> None:
        self.disconnect()
        self.ser = serial.Serial(port, baudrate=baud, timeout=0.2)
        time.sleep(2.0)  # Uno reset window
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self.ser is not None
        while not self._stop.is_set():
            try:
                line = self.ser.readline().decode("utf-8", errors="ignore").strip()
            except Exception:
                line = ""

            if not line or not line.startswith("DIST:"):
                continue

            try:
                value = float(line.split(":", 1)[1].strip())
            except Exception:
                continue

            self.latest_distance = max(0.0, min(value, 1000.0))
            self.last_update_ts = time.time()

    def disconnect(self) -> None:
        self._stop.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.ser is not None:
            try:
                self.ser.close()
            except Exception:
                pass
        self.ser = None

    @property
    def is_connected(self) -> bool:
        return self.ser is not None


class DataCollectorGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Rover Training Data Collector")
        self.root.geometry("1100x760")

        self.reader = DistanceReader()
        self.samples: list[Sample] = []

        self.port_var = tk.StringVar(value="/dev/ttyACM0")
        self.baud_var = tk.StringVar(value="9600")
        self.output_var = tk.StringVar(value=str(Path.cwd() / "training_data.csv"))
        self.status_var = tk.StringVar(value="Disconnected")
        self.latest_dist_var = tk.StringVar(value="Latest distance: --")
        self.count_var = tk.StringVar(value="Samples: 0")

        self.turn_hint_var = tk.IntVar(value=50)

        self._build_ui()
        self._load_existing()
        self._refresh_stats()
        self._draw_timeline()
        self._tick()

    def _build_ui(self) -> None:
        top = ttk.Frame(self.root, padding=10)
        top.pack(fill="x")

        ttk.Label(top, text="Port").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.port_var, width=20).grid(row=0, column=1, padx=4)

        ttk.Label(top, text="Baud").grid(row=0, column=2, sticky="w")
        ttk.Entry(top, textvariable=self.baud_var, width=10).grid(row=0, column=3, padx=4)

        ttk.Label(top, text="Output CSV").grid(row=0, column=4, sticky="w")
        ttk.Entry(top, textvariable=self.output_var, width=45).grid(row=0, column=5, padx=4)

        self.connect_btn = ttk.Button(top, text="Connect", command=self._toggle_connection)
        self.connect_btn.grid(row=0, column=6, padx=6)

        ttk.Label(top, textvariable=self.status_var).grid(row=0, column=7, padx=8)

        controls = ttk.Frame(self.root, padding=(10, 0, 10, 10))
        controls.pack(fill="x")

        ttk.Label(controls, textvariable=self.latest_dist_var, font=("Arial", 11, "bold")).pack(anchor="w")
        ttk.Label(controls, textvariable=self.count_var).pack(anchor="w")

        hint_row = ttk.Frame(controls)
        hint_row.pack(fill="x", pady=(8, 4))
        ttk.Label(hint_row, text="Turn Hint (0-100)").pack(side="left")
        ttk.Scale(hint_row, from_=0, to=100, variable=self.turn_hint_var, orient="horizontal").pack(
            side="left", fill="x", expand=True, padx=10
        )
        self.hint_label = ttk.Label(hint_row, text="50")
        self.hint_label.pack(side="left")

        btn_row = ttk.Frame(controls)
        btn_row.pack(fill="x", pady=6)
        ttk.Button(btn_row, text="Save LEFT (label=0)", command=lambda: self._save_sample(0)).pack(
            side="left", padx=4
        )
        ttk.Button(btn_row, text="Save RIGHT (label=1)", command=lambda: self._save_sample(1)).pack(
            side="left", padx=4
        )
        ttk.Button(btn_row, text="Reload CSV", command=self._load_existing).pack(side="left", padx=12)

        # Timeline canvas
        canvas_frame = ttk.LabelFrame(self.root, text="Timeline (all points in CSV)", padding=8)
        canvas_frame.pack(fill="both", expand=True, padx=10, pady=6)
        self.canvas = tk.Canvas(canvas_frame, bg="white", height=420)
        self.canvas.pack(fill="both", expand=True)

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _csv_path(self) -> Path:
        return Path(self.output_var.get()).expanduser().resolve()

    def _ensure_csv(self) -> None:
        path = self._csv_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["distance_cm", "turn_hint", "label"])

    def _load_existing(self) -> None:
        self.samples.clear()
        try:
            self._ensure_csv()
            path = self._csv_path()
            with path.open("r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        d = int(float(row["distance_cm"]))
                        h = int(float(row["turn_hint"]))
                        y = int(float(row["label"]))
                    except Exception:
                        continue
                    if y not in (0, 1):
                        continue
                    self.samples.append(Sample(distance_cm=max(0, min(100, d)), turn_hint=max(0, min(100, h)), label=y))
        except Exception as exc:
            messagebox.showerror("CSV Error", f"Failed to load CSV:\n{exc}")
        self._refresh_stats()
        self._draw_timeline()

    def _append_sample(self, s: Sample) -> None:
        self._ensure_csv()
        with self._csv_path().open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([s.distance_cm, s.turn_hint, s.label])

    def _toggle_connection(self) -> None:
        if self.reader.is_connected:
            self.reader.disconnect()
            self.status_var.set("Disconnected")
            self.connect_btn.config(text="Connect")
            return

        try:
            baud = int(self.baud_var.get())
            self.reader.connect(self.port_var.get().strip(), baud)
            self.status_var.set("Connected")
            self.connect_btn.config(text="Disconnect")
        except Exception as exc:
            messagebox.showerror("Serial Error", f"Failed to connect:\n{exc}")

    def _save_sample(self, label: int) -> None:
        dist = self.reader.latest_distance
        if dist is None:
            messagebox.showwarning("No Distance", "No DIST reading available yet from serial.")
            return

        hint = int(self.turn_hint_var.get())
        sample = Sample(distance_cm=int(round(max(0.0, min(100.0, dist)))), turn_hint=max(0, min(100, hint)), label=label)

        try:
            self._append_sample(sample)
            self.samples.append(sample)
            self._refresh_stats()
            self._draw_timeline()
        except Exception as exc:
            messagebox.showerror("Save Error", f"Failed to append sample:\n{exc}")

    def _refresh_stats(self) -> None:
        self.count_var.set(f"Samples: {len(self.samples)}")
        self.hint_label.config(text=str(int(self.turn_hint_var.get())))

    def _draw_timeline(self) -> None:
        c = self.canvas
        c.delete("all")

        w = max(200, c.winfo_width())
        h = max(200, c.winfo_height())
        left, right, top, bottom = 60, 20, 25, 45
        plot_w = w - left - right
        plot_h = h - top - bottom

        # Axes
        c.create_line(left, top, left, top + plot_h, fill="#333")
        c.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="#333")
        c.create_text(10, top + 6, text="100", anchor="w", fill="#555")
        c.create_text(18, top + plot_h - 2, text="0", anchor="w", fill="#555")
        c.create_text(left + plot_w - 2, h - 8, text="time ->", anchor="e", fill="#555")

        # Legend
        c.create_rectangle(w - 300, 8, w - 290, 18, fill="#2563eb", outline="")
        c.create_text(w - 285, 13, text="distance_cm", anchor="w", fill="#333")
        c.create_rectangle(w - 190, 8, w - 180, 18, fill="#f97316", outline="")
        c.create_text(w - 175, 13, text="turn_hint", anchor="w", fill="#333")
        c.create_rectangle(w - 95, 8, w - 85, 18, fill="#16a34a", outline="")
        c.create_text(w - 80, 13, text="label=1", anchor="w", fill="#333")
        c.create_rectangle(w - 95, 22, w - 85, 32, fill="#dc2626", outline="")
        c.create_text(w - 80, 27, text="label=0", anchor="w", fill="#333")

        if not self.samples:
            c.create_text(w / 2, h / 2, text="No samples yet. Save LEFT/RIGHT samples to build timeline.", fill="#777")
            return

        n = len(self.samples)

        def x_of(i: int) -> float:
            if n <= 1:
                return left
            return left + (i / (n - 1)) * plot_w

        def y_of(v: int) -> float:
            # map 0..100 to bottom..top
            return top + (1.0 - v / 100.0) * plot_h

        # Draw lines for distance and hint
        dist_pts = []
        hint_pts = []
        for i, s in enumerate(self.samples):
            x = x_of(i)
            dist_pts.extend([x, y_of(s.distance_cm)])
            hint_pts.extend([x, y_of(s.turn_hint)])

        if len(dist_pts) >= 4:
            c.create_line(*dist_pts, fill="#2563eb", width=2, smooth=False)
        if len(hint_pts) >= 4:
            c.create_line(*hint_pts, fill="#f97316", width=2, smooth=False)

        # Label dots near bottom strip
        y_label = top + plot_h + 12
        for i, s in enumerate(self.samples):
            x = x_of(i)
            color = "#16a34a" if s.label == 1 else "#dc2626"
            c.create_oval(x - 2.5, y_label - 2.5, x + 2.5, y_label + 2.5, fill=color, outline=color)

        c.create_text(left, top - 10, text=f"Total points: {n}", anchor="w", fill="#444")

    def _tick(self) -> None:
        # update live distance label + slider label + redraw only if size changed
        d = self.reader.latest_distance
        if d is None:
            self.latest_dist_var.set("Latest distance: --")
        else:
            age = "?"
            if self.reader.last_update_ts is not None:
                age = f"{(time.time() - self.reader.last_update_ts):.2f}s"
            self.latest_dist_var.set(f"Latest distance: {d:.1f} cm (age {age})")

        self.hint_label.config(text=str(int(self.turn_hint_var.get())))
        self.root.after(120, self._tick)

    def _on_close(self) -> None:
        self.reader.disconnect()
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    app = DataCollectorGUI(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
