"""laya helpers for the laya backend: the config, the temperature, the cut texts and the hold-out split."""

from typing import Any

from laya.common import QTYPES, clamp_temperature, encode_text, serialize_state, state_room, temp_bucket
from laya.train import TrainConfig, to_internal

from metis.data.models import AUGMENTED, Row
from metis.data.split import group_of, held_keys

from .spec import QUESTION_ID, LayaOptions, noul_temperature

NOUL: int = QTYPES["noul"]
BUCKET: str = temp_bucket(NOUL, 2)


def temperature_of(cfg: dict[str, object]) -> float:
    """Return the noul temperature of a config, clamped as laya clamps it at inference."""
    return clamp_temperature(t=noul_temperature(cfg=cfg, bucket=BUCKET, noul=NOUL))


def train_config(options: LayaOptions, epochs: int, seed: int) -> TrainConfig:
    """Return laya's training config. loss=rlcd turns on laya's policy-gradient objective."""
    return TrainConfig(
        epochs=epochs,
        micro_batch=options.micro_batch,
        grad_accum=options.grad_accum,
        encoder_lr=options.encoder_lr,
        head_lr=options.head_lr,
        loss=options.loss,
        rl_samples=options.rl_samples,
        sigma_start=options.sigma_start,
        sigma_end=options.sigma_end,
        freeze_encoder=options.freeze_encoder,
        seed=seed,
        log_every=options.log_every,
        gradient_checkpointing=options.gradient_checkpointing,
    )


def truncated(
    tok: Any, question: dict[str, object], max_len: int, head_max_len: int, texts: list[str], max_share: float
) -> int:
    """Count the texts that do not fit next to the question in max_len tokens, so laya cuts their end.

    Raise an error when more than max_share of the texts are cut.
    """
    q: dict[str, Any] = to_internal(qid=QUESTION_ID, question=question)
    budget: int = state_room(tok=tok, q=q, max_len=max_len, head_max_len=head_max_len)
    states: list[str] = [serialize_state({"prompt": text}).replace(tok.mask_token, " ") for text in texts]
    sizes: list[int] = [len(encode_text(tok, state, add_special_tokens=False)["input_ids"]) for state in states]
    cut: int = sum(1 for size in sizes if size > budget)
    if cut > max_share * len(texts):
        raise ValueError(f"{cut} of {len(texts)} rows exceed max_len={max_len}; raise max_truncated to accept")
    return cut


def hold_out(rows: list[Row], seed: int) -> tuple[list[Row], list[Row]]:
    """Return the rows to fit and the rows to calibrate on.

    The calibration rows are whole goal groups: a tenth of the groups, at most 400 rows, and no augmented kinds.
    """
    keys: list[str] = [group_of(row=row) for row in rows if row.kind not in AUGMENTED]
    held: set[str] = held_keys(keys=keys, seed=seed, share=0.1, max_rows=400)
    fitting: list[Row] = [row for row in rows if group_of(row=row) not in held]
    return fitting, [row for row in rows if group_of(row=row) in held and row.kind not in AUGMENTED]
