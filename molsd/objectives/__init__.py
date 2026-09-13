from molsd.objectives.base import Objective, TrainingPair
from molsd.objectives.epsilon import EpsilonObjective
from molsd.objectives.flow_matching import FlowMatchingObjective


def build_objective(cfg) -> Objective:
    """cfg: the `objective` config section."""
    if cfg.type == "flow_matching":
        return FlowMatchingObjective(
            timestep_sampling=cfg.timestep_sampling,
            logit_mean=cfg.logit_mean,
            logit_std=cfg.logit_std,
            shift=cfg.shift,
        )
    if cfg.type == "epsilon":
        return EpsilonObjective()
    raise ValueError(f"unknown objective type: {cfg.type}")


__all__ = ["Objective", "TrainingPair", "EpsilonObjective", "FlowMatchingObjective", "build_objective"]
