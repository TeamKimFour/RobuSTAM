"""Walk-forward 분할 — fold 경계 생성 (expanding · rolling 두 모드).

config `split` 블록으로 각 fold의 train/valid/test 거래일 경계를 만든다. test 블록은
공통으로 `test_blocks` 목록을 전진하며, train↔test 사이에 embargo(거래일) 갭을 둬 경계
누수를 막는다. 경계는 **거래일 인덱스** 기준(캘린더일 아님).

두 모드(`split.mode`):
  - `expanding`(기본) — anchor(예: 2010)를 고정하고 train을 fold마다 계속 누적한다.
    `split.anchor_start` 필요.
  - `rolling` — train_all 길이를 `split.rolling_train_days`(거래일)로 고정하고, test
    직전부터 그만큼만 뒤로 잘라 쓴다. fold마다 train 크기가 같아 비교가 깔끔하고, 오래된
    국면(예: 2010년대 저금리)이 최근 국면 예측에 계속 섞이는 것을 막는다. `split.anchor_start`는
    무시한다.

두 모드 모두 train_all(=train+valid) 꼬리 `valid_days`를 validation으로 떼어내는 방식은 동일하다.
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
    """가용 거래일 인덱스에서 walk-forward fold 목록을 만든다 (expanding 또는 rolling)."""
    split = cfg["split"]
    idx = pd.DatetimeIndex(available_index).sort_values()
    embargo = int(split["embargo_days"])
    valid_days = int(split["valid_days"])
    mode = split.get("mode", "expanding")
    if mode not in ("expanding", "rolling"):
        raise ValueError(f"split.mode는 'expanding' 또는 'rolling'이어야 합니다: {mode!r}")

    anchor = pd.Timestamp(split["anchor_start"]) if mode == "expanding" else None
    rolling_train_days = int(split["rolling_train_days"]) if mode == "rolling" else None

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

        if mode == "rolling":
            # train_all 길이를 rolling_train_days로 고정 — anchor 무시, test 직전부터 뒤로 자른다.
            train_start_pos = train_end_pos - rolling_train_days + 1
            if train_start_pos < 0:
                raise ValueError(
                    f"fold {i}: rolling_train_days({rolling_train_days})가 가용 데이터보다 "
                    f"깁니다(train_end_pos={train_end_pos})"
                )
            train_all = idx[train_start_pos : train_end_pos + 1]
        else:
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
