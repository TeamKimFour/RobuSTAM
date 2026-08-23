"""Walk-forward 분할(split) 재구성 후보 비교 — 분포 드리프트·fold 크기를 RL 없이 스크리닝.

`docs/model_diagnosis.md` §3이 진단한 "train→test 분포 이동(최대 22σ)"과 §6이 남긴 미해결
질문("분할 구조 자체가 문제인가")에 대한 후속이다. RL 재학습은 비싸므로(200k timesteps ×
fold × 콤보), 분할 재구성 후보들을 **데이터 레벨에서** 먼저 값싸게 비교한다 — feature combo를
`screen_combos.py`로 먼저 거른 것과 같은 철학.

비교 축(민지 6주차 과제 "데이터 재구성"):
  - 분할 방식: expanding(현행) vs rolling(신규, `splits.rolling_train_days`)
  - fold 개수/구간: 현행 2년 단위 3개 vs 연 단위 세분화
  - train 기간 길이: expanding(무제한 누적) vs rolling 고정폭(5년·8년)

`config.yaml`의 `split`(SSOT)은 건드리지 않는다 — 여기 정의된 후보는 비교용 사본이다.
채택 시 별도로 `config.yaml`·`docs/data_pipeline.md` §4-1을 갱신해야 한다(CLAUDE.md 팀 합의 필요).

실행: `python -m src.data.screen_splits`
"""

from __future__ import annotations

import pandas as pd

from src.config_loader import get_window, load_config, resolve_combo
from src.data.assemble import assemble_state_matrix
from src.data.collect import load_raw
from src.data.features import compute_features
from src.data.normalize import ZScoreScaler
from src.data.returns import log_returns
from src.data.splits import make_folds

TRADING_DAYS_PER_YEAR = 252


def _annual_test_blocks(start_year: int, end_year: int) -> list[list[str]]:
    """연 단위 test_blocks(캘린더 연도, 각 1/1~12/31)를 만든다."""
    return [[f"{y}-01-01", f"{y}-12-31"] for y in range(start_year, end_year + 1)]


# 비교 후보 — config.yaml 기본값(anchor 2010, embargo 34, valid_days 252)은 그대로 두고
# split 섹션만 갈아끼운다. 이름은 report에 그대로 노출된다.
CANDIDATE_SPLITS: dict[str, dict] = {
    "A_baseline_expanding_2y": {
        "mode": "expanding",
        "anchor_start": "2010-01-01",
        "test_blocks": [
            ["2020-01-01", "2021-12-31"],
            ["2022-01-01", "2023-12-31"],
            ["2024-01-01", "2025-12-31"],
        ],
        "valid_days": 252,
        "embargo_days": 34,
    },
    "B_rolling_8y_2y_blocks": {
        "mode": "rolling",
        "rolling_train_days": 8 * TRADING_DAYS_PER_YEAR,
        "test_blocks": [
            ["2020-01-01", "2021-12-31"],
            ["2022-01-01", "2023-12-31"],
            ["2024-01-01", "2025-12-31"],
        ],
        "valid_days": 252,
        "embargo_days": 34,
    },
    "C_rolling_5y_2y_blocks": {
        "mode": "rolling",
        "rolling_train_days": 5 * TRADING_DAYS_PER_YEAR,
        "test_blocks": [
            ["2020-01-01", "2021-12-31"],
            ["2022-01-01", "2023-12-31"],
            ["2024-01-01", "2025-12-31"],
        ],
        "valid_days": 252,
        "embargo_days": 34,
    },
    "D_expanding_annual": {
        "mode": "expanding",
        "anchor_start": "2010-01-01",
        "test_blocks": _annual_test_blocks(2019, 2025),
        "valid_days": 252,
        "embargo_days": 34,
    },
    "E_rolling_8y_annual": {
        "mode": "rolling",
        "rolling_train_days": 8 * TRADING_DAYS_PER_YEAR,
        "test_blocks": _annual_test_blocks(2019, 2025),
        "valid_days": 252,
        "embargo_days": 34,
    },
}


def _build_state_all(config_path: str = "config/config.yaml") -> tuple[pd.DataFrame, dict]:
    """실데이터로 (콤보 무관) 지표까지 조립된 비정규화 state_all을 한 번만 만든다.

    분할 후보 비교는 어느 콤보를 쓰든 상대적 결론(분포 드리프트가 줄어드는가)은 비슷하므로
    이미 실빌드된 `full`을 그대로 쓴다 — 콤보별로 반복 계산할 필요 없음.
    """
    cfg = resolve_combo(load_config(config_path), "full")
    prices = load_raw(cfg["data"]["raw_dir"])
    logret = log_returns(prices)
    close = prices.loc[logret.index]
    asset_feat, market_feat = compute_features(close, logret, cfg)
    state_all = assemble_state_matrix(logret, asset_feat, market_feat, cfg)
    return state_all, cfg


def _fold_drift(train_raw: pd.DataFrame, test_raw: pd.DataFrame, cfg: dict, W: int) -> dict:
    """train에서만 fit한 스케일러로 test를 transform해 분포 드리프트 통계를 낸다.

    model_diagnosis.md §3과 동일 정의: mean|test_mean|·mean|test_std-1|·max|z|.
    """
    scaler = ZScoreScaler.from_config(cfg).fit(train_raw, W)
    cols = [c for c in train_raw.columns if c.startswith(("feat_", "mkt_"))]
    test_norm = scaler.transform(test_raw)[cols]
    if len(test_norm) == 0 or not cols:
        return {"mean_abs_test_mean": float("nan"), "mean_abs_test_std_m1": float("nan"),
                "max_abs_z": float("nan")}
    return {
        "mean_abs_test_mean": float(test_norm.mean().abs().mean()),
        "mean_abs_test_std_m1": float((test_norm.std() - 1).abs().mean()),
        "max_abs_z": float(test_norm.abs().to_numpy().max()),
    }


def evaluate_candidate(state_all: pd.DataFrame, base_cfg: dict, split_cfg: dict) -> pd.DataFrame:
    """후보 split 하나를 실제 state_all에 적용해 fold별 크기·분포드리프트 표를 낸다."""
    cfg = {**base_cfg, "split": split_cfg}
    W = get_window(cfg)
    folds = make_folds(state_all.index, cfg)

    rows = []
    for fold in folds:
        train_raw = state_all.loc[fold.train_start : fold.train_end]
        valid_raw = state_all.loc[fold.valid_start : fold.valid_end]
        test_raw = state_all.loc[fold.test_start : fold.test_end]
        drift = _fold_drift(train_raw, test_raw, cfg, W)
        rows.append({
            "fold_id": fold.fold_id,
            "train_start": fold.train_start.date(), "train_end": fold.train_end.date(),
            "test_start": fold.test_start.date(), "test_end": fold.test_end.date(),
            "n_train": len(train_raw), "n_valid": len(valid_raw), "n_test": len(test_raw),
            **drift,
        })
    return pd.DataFrame(rows)


def compare_all(config_path: str = "config/config.yaml") -> dict[str, pd.DataFrame]:
    """CANDIDATE_SPLITS 전체를 평가해 {후보명: fold별 표} 딕셔너리로 반환한다."""
    state_all, base_cfg = _build_state_all(config_path)
    return {name: evaluate_candidate(state_all, base_cfg, split_cfg)
            for name, split_cfg in CANDIDATE_SPLITS.items()}


def summarize(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """후보별 요약 — fold 개수·평균 train 크기·train 크기 변동폭·평균/최대 드리프트."""
    rows = {}
    for name, df in results.items():
        rows[name] = {
            "n_folds": len(df),
            "mean_n_train": df["n_train"].mean(),
            "train_size_range": df["n_train"].max() - df["n_train"].min(),
            "mean_max_abs_z": df["max_abs_z"].mean(),
            "worst_max_abs_z": df["max_abs_z"].max(),
            "mean_drift_mean": df["mean_abs_test_mean"].mean(),
        }
    return pd.DataFrame(rows).T


def main() -> None:
    pd.set_option("display.width", 140)
    pd.set_option("display.max_columns", 20)
    print("=" * 100)
    print("분할(split) 재구성 후보 비교 — 분포 드리프트·fold 크기 (RL 없이 데이터 레벨 스크리닝)")
    print("=" * 100)
    results = compare_all()
    for name, df in results.items():
        print(f"\n--- {name} ---")
        print(df.to_string(index=False))

    print("\n" + "=" * 100)
    print("요약 (전체 후보 비교)")
    print("=" * 100)
    print(summarize(results).to_string())
    print(
        "\n  · mean_max_abs_z: fold별 최대 |z| 평균 — 낮을수록 train→test 분포 이동이 작음"
        "\n  · train_size_range: fold간 train 크기 차이 — 0이면 전 fold 동일 크기(rolling 특징)"
        "\n  · n_folds: 많을수록 OOS 통계적 검정력↑, 대신 fold당 test 기간이 짧아짐(연단위 트레이드오프)"
        "\n  ⚠️ 이 비교는 RL 성능 예측이 아니라 분포 안정성 사전 필터다 — 최종 판단은 후보를 실제로"
        "\n     골라 도현 RL로 재학습해봐야 한다."
    )


if __name__ == "__main__":
    main()
