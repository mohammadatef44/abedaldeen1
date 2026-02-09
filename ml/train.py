from __future__ import annotations

import json
from pathlib import Path

from ml.model import METRICS_PATH, train_and_evaluate


def main() -> None:
    metrics = train_and_evaluate()
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "accuracy": round(metrics.accuracy, 3),
        "report": metrics.report,
    }
    METRICS_PATH.write_text(json.dumps(payload, indent=2))
    print(f"Saved metrics to {METRICS_PATH}")


if __name__ == "__main__":
    main()
