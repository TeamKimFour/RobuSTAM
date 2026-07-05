"""Expanding walk-forward 분할 — fold 경계 생성.

config `split` 블록으로 각 fold의 train/valid/test 거래일 경계를 만든다. anchor(2010)를 고정하고
train을 누적(expanding)하며, test 블록을 2년 단위로 전진한다. train↔test 사이에 embargo(거래일) 갭을
둬 경계 누수를 막는다. 경계는 **거래일 인덱스** 기준(캘린더일 아님).
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class Fold:
    fold_id: int
    mode: str
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_days: int

    def to_meta_dict(self) -> dict:
        """feature_store.write_fold 규격(문자열 날짜)."""
        return {
            "fold_id": self.fold_id,
            "mode": self.mode,
            "train_start": str(self.train_start.date()),
            "train_end": str(self.train_end.date()),
            "valid_start": str(self.valid_start.date()),
            "valid_end": str(self.valid_end.date()),
            "test_start": str(self.test_start.date()),
            "test_end": str(self.test_end.date()),
            "embargo_days": self.embargo_days,
        }


def make_folds(available_index: pd.DatetimeIndex, cfg: dict) -> list[Fold]:
    """가용 거래일 인덱스에서 Expanding walk-forward fold 목록을 만든다."""
    split = cfg["split"]
    idx = pd.DatetimeIndex(available_index).sort_values()
    anchor = pd.Timestamp(split["anchor_start"])
    embargo = int(split["embargo_days"])
    valid_days = int(split["valid_days"])
    mode = split.get("mode", "expanding")

    folds: list[Fold] = []
    for i, (tb_start, tb_end) in enumerate(split["test_blocks"], start=1):
        tb_start, tb_end = pd.Timestamp(tb_start), pd.Timestamp(tb_end)
        test_idx = idx[(idx >= tb_start) & (idx <= tb_end)]
        if len(test_idx) == 0:
            continue

        # test 시작보다 embargo 거래일 이전을 train 끝으로 (경계 누수 차단)
        test_first_pos = idx.get_loc(test_idx[0])
        train_end_pos = test_first_pos - 1 - embargo
        if train_end_pos < 0:
            raise ValueError(f"fold {i}: embargo 적용 후 train 구간이 없습니다")

        train_all = idx[(idx >= anchor) & (idx <= idx[train_end_pos])]
        if len(train_all) <= valid_days:
            raise ValueError(
                f"fold {i}: train_all({len(train_all)}) <= valid_days({valid_days})"
            )

        valid_idx = train_all[-valid_days:]
        train_idx = train_all[:-valid_days]

        folds.append(
            Fold(
                fold_id=i,
                mode=mode,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                valid_start=valid_idx[0],
                valid_end=valid_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                embargo_days=embargo,
            )
        )
    return folds
