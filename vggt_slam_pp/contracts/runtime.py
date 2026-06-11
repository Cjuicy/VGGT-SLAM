"""Immutable identity for a baseline or front-end bridge run."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SolverMode = Literal["baseline_sl4", "baseline_sim3_compat"]
RunPurpose = Literal["baseline_reference", "pp_frontend_bridge"]


class RunIdentity(BaseModel):
    """Parameters that distinguish scientifically different M0 runs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    solver_mode: SolverMode
    run_purpose: RunPurpose
    max_loops: int = Field(ge=0)
    submap_size: int = Field(gt=0)
    min_disparity: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_bridge_mode(self) -> "RunIdentity":
        """Prevent baseline back-end choices from leaking into ++ bridge data."""
        if self.run_purpose == "pp_frontend_bridge":
            if self.solver_mode != "baseline_sim3_compat":
                raise ValueError(
                    "pp_frontend_bridge requires baseline_sim3_compat"
                )
            if self.max_loops != 0:
                raise ValueError("pp_frontend_bridge requires max_loops=0")
        return self
