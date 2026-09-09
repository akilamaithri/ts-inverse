"""File-based logger for offline runs (no wandb account, no network).

Enabled by putting "local" in a config's ``logger_service``. Writes one
directory per run under ``TS_INVERSE_RESULTS_DIR`` (default ``./results``):

    <experiment_name>/<run_name>/
        config.json      final merged config
        metrics.jsonl    one JSON object per logged step
        summary.json     last value of every scalar metric + wall time
        series_*.csv     original-vs-reconstructed series dataframes
        figure_*.png     matplotlib figures (only if TS_INVERSE_SAVE_FIGS=1)
"""

import json
import os
import time

import numpy as np


def _jsonable(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


class LocalLogger:
    def __init__(self, run_name, config):
        root = os.environ.get("TS_INVERSE_RESULTS_DIR", "./results")
        # A stable, caller-supplied label keeps runs resumable; fall back to the
        # timestamped name the other logger backends use.
        run_name = str(config.get("run_label") or run_name)
        self.dir = os.path.join(root, str(config.get("experiment_name", "default")), run_name)
        os.makedirs(self.dir, exist_ok=True)
        self.save_figures = os.environ.get("TS_INVERSE_SAVE_FIGS", "0") == "1"
        self.started_at = time.time()
        self.summary = {}
        self.metrics_path = os.path.join(self.dir, "metrics.jsonl")
        open(self.metrics_path, "w").close()
        self.write_config(config)

    def write_config(self, config):
        with open(os.path.join(self.dir, "config.json"), "w") as f:
            json.dump(_jsonable(dict(config)), f, indent=2, sort_keys=True)

    def log_metrics(self, metrics_dict, step):
        record = {"step": int(step)}
        for key, value in metrics_dict.items():
            record[key] = _jsonable(value)
            if isinstance(value, (int, float, np.floating, np.integer)) and not isinstance(value, bool):
                self.summary[key] = float(value)
        with open(self.metrics_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_dataframe(self, df, step, log_name=""):
        df.to_csv(os.path.join(self.dir, f"series{log_name}_step{int(step)}.csv"), index=False)

    def log_figure(self, fig, step, log_name=""):
        if self.save_figures:
            fig.savefig(os.path.join(self.dir, f"figure{log_name}_step{int(step)}.png"), dpi=90, bbox_inches="tight")

    def finish(self):
        self.summary["wall_time_seconds"] = time.time() - self.started_at
        with open(os.path.join(self.dir, "summary.json"), "w") as f:
            json.dump(_jsonable(self.summary), f, indent=2, sort_keys=True)
