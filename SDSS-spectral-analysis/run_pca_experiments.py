"""Run only the notebook's PCA retuning/timing section independently of SSH."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import json
import os
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUTPUT = ROOT / "data/dr17_5000/pca/efficiency"
OUTPUT.mkdir(parents=True, exist_ok=True)
STATUS = {"state": "starting", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat()}
STARTED = time.monotonic()


def status(state, message):
    STATUS.update(state=state, last_message=message,
                  elapsed_seconds=round(time.monotonic() - STARTED, 1))
    temporary = OUTPUT / "background_status.json.part"
    temporary.write_text(json.dumps(STATUS, indent=2), encoding="utf-8")
    temporary.replace(OUTPUT / "background_status.json")


def execute():
    notebook = nbformat.read(ROOT / "project1.ipynb", as_version=4)
    start = next(i for i, cell in enumerate(notebook.cells)
                 if cell.cell_type == "markdown" and cell.source.startswith(
                     "## Retune the two-component random forest and compare efficiency"))
    end = next((i for i in range(start + 1, len(notebook.cells))
                if notebook.cells[i].cell_type == "markdown" and notebook.cells[i].source.startswith("## ")),
               len(notebook.cells))
    part = nbformat.v4.new_notebook(
        cells=[cell for cell in notebook.cells[start:end] if cell.cell_type == "code"],
        metadata=notebook.metadata,
    )

    def save(**kwargs):
        latest = nbformat.read(ROOT / "project1.ipynb", as_version=4)
        results = {cell.id: cell for cell in part.cells}
        for cell in latest.cells:
            result = results.get(cell.id)
            if result is not None and result.source == cell.source:
                cell.outputs = result.outputs
                cell.execution_count = result.execution_count
                cell.metadata.update(result.metadata)
        nbformat.validate(latest)
        nbformat.write(latest, ROOT / "project1.ipynb")

    class ProgressClient(NotebookClient):
        def process_message(self, message, cell, cell_index):
            if message.get("msg_type") == "stream":
                message_text = message["content"]["text"]
                print(message_text, end="", flush=True)
                lines = [line.strip() for line in message_text.splitlines() if line.strip()]
                if lines:
                    status("running", lines[-1])
            return super().process_message(message, cell, cell_index)

    try:
        ProgressClient(part, timeout=None, kernel_name="sdss-spectral-analysis",
                       on_cell_executed=save).execute()
    finally:
        save()


if __name__ == "__main__":
    if os.name == "nt":
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with (OUTPUT / "background.log").open("a", encoding="utf-8", buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                status("running", "Starting PCA retuning and timing comparisons.")
                try:
                    execute()
                    status("complete", "PCA retuning and all 20 timing comparisons completed.")
                except BaseException as exc:
                    traceback.print_exc()
                    status("failed", f"{type(exc).__name__}: {exc}")
                    raise
    finally:
        if os.name == "nt":
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
