from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from mmengine.hooks import Hook


def _unwrap_model(model):
    # mmengine's model wrappers typically expose `.module`
    if hasattr(model, "module"):
        return model.module
    return model


@dataclass(frozen=True)
class DistillSchedule:
    target: float
    start_iter: int = 0
    warmup_iters: int = 0
    ramp_iters: int = 0
    # Optional early-stop / decay:
    decay_start_iter: Optional[int] = None
    decay_iters: int = 0
    end_iter: Optional[int] = None  # if set, after end_iter => 0


def _linear01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return float(x)


def compute_distill_weight(cur_iter: int, sched: DistillSchedule) -> float:
    """
    Piecewise schedule:
      - [start, start+warmup): 0
      - [start+warmup, start+warmup+ramp): linear 0->target
      - hold at target
      - optional decay: from decay_start_iter for decay_iters linearly target->0
      - optional end_iter: force 0 after end_iter
    """
    t = float(sched.target)
    if t <= 0.0:
        return 0.0

    it = int(cur_iter)
    if it < int(sched.start_iter):
        return 0.0

    warm_end = int(sched.start_iter) + int(sched.warmup_iters)
    ramp_end = warm_end + int(sched.ramp_iters)

    if it < warm_end:
        w = 0.0
    elif int(sched.ramp_iters) > 0 and it < ramp_end:
        frac = (it - warm_end) / float(max(1, sched.ramp_iters))
        w = t * _linear01(frac)
    else:
        w = t

    # decay
    if sched.decay_start_iter is not None and int(sched.decay_iters) > 0:
        ds = int(sched.decay_start_iter)
        de = ds + int(sched.decay_iters)
        if it >= ds:
            frac = (it - ds) / float(max(1, sched.decay_iters))
            w = max(0.0, t * (1.0 - _linear01(frac)))
        if it >= de:
            w = 0.0

    # hard end
    if sched.end_iter is not None and it >= int(sched.end_iter):
        w = 0.0

    return float(w)


class DistillWeightSchedulerHook(Hook):
    """
    Update `model.distill_weight` each train iteration.

    Why:
      - Avoid applying strong distillation from iter=0.
      - Let you test "warmup+ramp+stop" without touching forward().

    Usage in config (example):

      from projects.main.hooks import DistillWeightSchedulerHook
      custom_hooks = [
          dict(
              type=DistillWeightSchedulerHook,
              target=0.02,
              warmup_iters=200,
              ramp_iters=300,
              decay_start_iter=800,
              decay_iters=200,
          )
      ]
    """

    def __init__(
        self,
        target: float,
        start_iter: int = 0,
        warmup_iters: int = 0,
        ramp_iters: int = 0,
        decay_start_iter: Optional[int] = None,
        decay_iters: int = 0,
        end_iter: Optional[int] = None,
        log_interval: int = 50,
    ) -> None:
        self.sched = DistillSchedule(
            target=float(target),
            start_iter=int(start_iter),
            warmup_iters=int(warmup_iters),
            ramp_iters=int(ramp_iters),
            decay_start_iter=int(decay_start_iter) if decay_start_iter is not None else None,
            decay_iters=int(decay_iters),
            end_iter=int(end_iter) if end_iter is not None else None,
        )
        self.log_interval = int(log_interval)

    def before_train_iter(self, runner, batch_idx: int, data_batch=None) -> None:
        model = _unwrap_model(runner.model)
        if not hasattr(model, "distill_weight"):
            return
        cur_iter = int(getattr(runner, "iter", 0))
        w = compute_distill_weight(cur_iter, self.sched)
        model.distill_weight = float(w)

        # lightweight logging on rank0
        if self.log_interval > 0 and cur_iter % self.log_interval == 0:
            try:
                rank0 = True
                if hasattr(runner, "rank"):
                    rank0 = int(getattr(runner, "rank")) == 0
                if rank0 and hasattr(runner, "logger") and runner.logger is not None:
                    runner.logger.info(f"[distill-sched] iter={cur_iter} distill_weight={w:.6f} target={self.sched.target:.6f}")
            except Exception:
                pass


