"""Run the notebook's forest search independently of an SSH/notebook connection."""

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timezone
from pathlib import Path
import ctypes
import json
import os
import sys
import time
import traceback

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
OUTPUT = ROOT / 'data' / 'dr17_5000' / 'random_forest_tuning'
OUTPUT.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUTPUT / 'background.log'
STATUS_PATH = OUTPUT / 'background_status.json'
STARTED = datetime.now(timezone.utc).isoformat()
START_CLOCK = time.monotonic()
STATE = {'state': 'starting', 'pid': os.getpid(), 'started_utc': STARTED,
         'log_file': str(LOG_PATH), 'last_message': ''}


def update_status(state=None, message=None):
    if state is not None:
        STATE['state'] = state
    if message is not None:
        STATE['last_message'] = message
    STATE['updated_utc'] = datetime.now(timezone.utc).isoformat()
    STATE['elapsed_seconds'] = round(time.monotonic() - START_CLOCK, 1)
    temporary = STATUS_PATH.with_suffix('.json.part')
    temporary.write_text(json.dumps(STATE, indent=2), encoding='utf-8')
    temporary.replace(STATUS_PATH)


def save_notebook_outputs(original, indices, part):
    path = ROOT / 'project1.ipynb'
    current = nbformat.read(path, as_version=4)
    for original_index, executed in zip(indices, part.cells[1:]):
        source_cell = original.cells[original_index]
        matches = [cell for cell in current.cells
                   if cell.get('id') == source_cell.get('id')]
        if len(matches) == 1 and matches[0].source == executed.source:
            matches[0].outputs = executed.outputs
            matches[0].execution_count = executed.execution_count
    nbformat.validate(current)
    nbformat.write(current, path)


def run():
    nb = nbformat.read(ROOT / 'project1.ipynb', as_version=4)
    start = next(i for i, cell in enumerate(nb.cells)
                 if cell.cell_type == 'markdown'
                 and cell.source.startswith('# Random Forest Regression'))
    end = next((i for i in range(start + 1, len(nb.cells))
                if nb.cells[i].cell_type == 'markdown'
                and nb.cells[i].source.startswith('# ')), len(nb.cells))
    indices = [i for i in range(start, end) if nb.cells[i].cell_type == 'code']
    setup = nbformat.v4.new_code_cell('''from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
from threadpoolctl import threadpool_limits
thread_limits = threadpool_limits(limits=1)
with np.load("data/dr17_5000/prepared_baseline/regression_inputs.npz", allow_pickle=False) as data:
    names = data["feature_names"]
    targets = data["target_names"]
    train_index = pd.Index(data["train_ids"], name="specObjID")
    val_index = pd.Index(data["validation_ids"], name="specObjID")
    X_train = pd.DataFrame(data["X_train"], index=train_index, columns=names)
    y_train = pd.DataFrame(data["y_train"], index=train_index, columns=targets)
    X_validation = pd.DataFrame(data["X_validation"], index=val_index, columns=names)
    y_validation = pd.DataFrame(data["y_validation"], index=val_index, columns=targets)
''')
    part = nbformat.v4.new_notebook(
        cells=[setup] + [nb.cells[i] for i in indices], metadata=nb.metadata)

    class ProgressClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            if msg.get('msg_type') == 'stream':
                text = msg['content']['text']
                print(text, end='', flush=True)
                lines = [line.strip() for line in text.splitlines() if line.strip()]
                if lines:
                    update_status('running', lines[-1])
            return super().process_message(msg, cell, cell_index)

    def save_stage_outputs(**kwargs):
        save_notebook_outputs(nb, indices, part)

    client = ProgressClient(
        part, timeout=None, kernel_name='sdss-spectral-analysis',
        on_cell_executed=save_stage_outputs)
    try:
        client.execute()
    finally:
        save_notebook_outputs(nb, indices, part)


def main():
    # Keep this Windows host awake during this job without changing global settings.
    if os.name == 'nt':
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        with LOG_PATH.open('a', encoding='utf-8', buffering=1) as log:
            with redirect_stdout(log), redirect_stderr(log):
                print(f'\n=== Random forest background run started {STARTED} ===', flush=True)
                update_status('running', 'Starting notebook kernel; completed trials will be reused.')
                try:
                    run()
                except BaseException as exc:
                    traceback.print_exc()
                    update_status('failed', f'{type(exc).__name__}: {exc}')
                    return 1
                print('All three tuning stages are complete for all four targets.', flush=True)
                update_status('complete', 'All three tuning stages completed for all four targets.')
                return 0
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    sys.exit(main())
