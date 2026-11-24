"""The laya backend parts that need no laya install: options, the question, the case format and probabilities."""

import math
from dataclasses import dataclass, fields
from typing import TypeVar

T = TypeVar("T")

QUESTION_ID: str = "prompt_injection"
LIMITS: dict[str, int] = {"max_len": 512, "head_max_len": 192}  # laya token limits when the checkpoint names none.


@dataclass(frozen=True)
class LayaOptions:
    """laya training choices. loss=rlcd is laya's policy-gradient objective on the labels.

    log_every prints laya's loss every that many micro-batches (0 = off): at micro_batch=8 and grad_accum=8, 50
    micro-batches are 400 rows, about 6 optimizer updates.
    """

    loss: str = "soft-ce"
    micro_batch: int = 8
    grad_accum: int = 4
    encoder_lr: float = 2.5e-5
    head_lr: float = 1e-4
    freeze_encoder: bool = False
    gradient_checkpointing: bool = False
    rl_samples: int = 4
    sigma_start: float = 0.4
    sigma_end: float = 0.1
    max_truncated: float = 0.01
    log_every: int = 50


def pick(options: dict[str, object], key: str, kind: type[T], default: T) -> T:
    """Return one option of the given type, or the default when it is missing."""
    value: object = options.get(key, default)
    if kind is float and isinstance(value, int) and not isinstance(value, bool):
        value = float(value)
    if type(value) is not kind or not isinstance(value, kind):
        raise ValueError(f"laya option {key} must be a {kind.__name__}, got {value!r}")
    return value


def read_options(options: dict[str, object]) -> LayaOptions:
    """Check the backend options of the settings and return them typed."""
    known: set[str] = {item.name for item in fields(LayaOptions)}
    unknown: list[str] = sorted(set(options) - known)
    if unknown:
        raise ValueError(f"unknown laya options {unknown}; known: {sorted(known)}")
    base: LayaOptions = LayaOptions()
    chosen: LayaOptions = LayaOptions(
        loss=pick(options=options, key="loss", kind=str, default=base.loss),
        micro_batch=pick(options=options, key="micro_batch", kind=int, default=base.micro_batch),
        grad_accum=pick(options=options, key="grad_accum", kind=int, default=base.grad_accum),
        encoder_lr=pick(options=options, key="encoder_lr", kind=float, default=base.encoder_lr),
        head_lr=pick(options=options, key="head_lr", kind=float, default=base.head_lr),
        freeze_encoder=pick(options=options, key="freeze_encoder", kind=bool, default=base.freeze_encoder),
        gradient_checkpointing=pick(
            options=options, key="gradient_checkpointing", kind=bool, default=base.gradient_checkpointing
        ),
        rl_samples=pick(options=options, key="rl_samples", kind=int, default=base.rl_samples),
        sigma_start=pick(options=options, key="sigma_start", kind=float, default=base.sigma_start),
        sigma_end=pick(options=options, key="sigma_end", kind=float, default=base.sigma_end),
        max_truncated=pick(options=options, key="max_truncated", kind=float, default=base.max_truncated),
        log_every=pick(options=options, key="log_every", kind=int, default=base.log_every),
    )
    if chosen.loss not in ("soft-ce", "rlcd"):
        raise ValueError(f"laya option loss must be soft-ce or rlcd, got {chosen.loss!r}")
    if not 0.0 <= chosen.max_truncated < 1.0:
        raise ValueError(f"laya option max_truncated must be in [0, 1), got {chosen.max_truncated!r}")
    return chosen


def case(text: str, question: dict[str, object], label: bool) -> dict[str, object]:
    """Return one laya training case: the text as the prompt of the injection question, and its label."""
    return {"state": {"prompt": text}, "questions": {QUESTION_ID: question}, "expected": {QUESTION_ID: label}}


def true_probability(logits: list[float], temperature: float) -> float:
    """Return the probability of the true option (index 1) of a yes/no question at a temperature."""
    scaled: list[float] = [value / temperature for value in logits]
    top: float = max(scaled)
    weights: list[float] = [math.exp(value - top) for value in scaled]
    return weights[1] / sum(weights)


def noul_temperature(cfg: dict[str, object], bucket: str, noul: int) -> float:
    """Return the temperature that laya applies to a two-option yes/no question with this checkpoint config.

    laya uses the temperature of the option-count bucket when the config has one, else the scalar of the type.
    """
    scalars: object = cfg.get("temperature")
    buckets: object = cfg.get("temperature_by_options")
    if isinstance(buckets, dict) and isinstance(buckets.get(bucket), (int, float)):
        return float(buckets[bucket])
    if isinstance(scalars, list) and len(scalars) > noul and isinstance(scalars[noul], (int, float)):
        return float(scalars[noul])
    return 1.0


def calibrated(cfg: dict[str, object], fitted: dict[str, object]) -> dict[str, object]:
    """Return the checkpoint config with the newly fitted temperatures in place of the base ones."""
    found: dict[str, object] = {key: item for key, item in cfg.items() if key != "temperature_by_options"}
    found.update(fine_tuned=True, temperature=fitted["temperature"])
    if fitted.get("temperature_by_options"):
        found["temperature_by_options"] = fitted["temperature_by_options"]
    return found
