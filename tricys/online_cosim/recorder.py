from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from tricys.online_cosim.schema import TrackResult, UnifiedStateVector


@dataclass(slots=True)
class InMemoryStepRecorder:
    """Store step-wise online co-simulation traces in memory."""

    input_rows: list[dict[str, Any]] = field(default_factory=list)
    output_rows: list[dict[str, Any]] = field(default_factory=list)

    def record_step(
        self,
        request: UnifiedStateVector,
        results: list[TrackResult],
    ) -> None:
        self.input_rows.append(
            {
                "component_name": request.component_name,
                "step_id": request.step_id,
                "seq_id": request.seq_id,
                "boundary_inputs": dict(request.boundary_inputs),
                "extra_state": (
                    dict(request.extra_state)
                    if request.extra_state is not None
                    else None
                ),
            }
        )

        for result_index, result in enumerate(results):
            self.output_rows.append(
                {
                    "component_name": request.component_name,
                    "step_id": request.step_id,
                    "seq_id": request.seq_id,
                    "result_index": result_index,
                    "outputs": dict(result.outputs),
                }
            )

    def to_dataframes(self) -> dict[str, pd.DataFrame]:
        return {
            "inputs": pd.DataFrame(self.input_rows),
            "outputs": pd.DataFrame(self.output_rows),
        }

    def clear(self) -> None:
        self.input_rows.clear()
        self.output_rows.clear()

    def export_csv(self, output_dir: str | Path) -> dict[str, Path]:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        frames = self.to_dataframes()
        target_paths = {
            "inputs": output_path / "online_inputs.csv",
            "outputs": output_path / "online_outputs.csv",
        }

        for key, frame in frames.items():
            frame.to_csv(target_paths[key], index=False)

        return target_paths
