"""Feature Store 빌드 오케스트레이터.

수집(캐시) → 로그수익률 → 지표 → 187 조립 → fold별 (train에서 정규화 fit → 전체 transform) →
Parquet 적재 + 익일수익률(targets) 저장 + 메타(folds·scaler_stats) 기록.

지표는 fold와 무관하게 1회만 계산(인과적)하고, **정규화만 fold별로 train에서 재fit**한다.
실행: `python -m src.data.build`
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from src.config_loader import get_assets, get_state_dim, get_window, load_config
from src.data import feature_store as fs
from src.data import schema
from src.data.assemble import assemble_state_matrix
from src.data.collect import load_raw
from src.data.features import compute_features
from src.data.normalize import ZScoreScaler
from src.data.returns import log_returns
from src.data.splits import make_folds

SPLITS = ("train", "valid", "test")


def build(config_path: str = "config/config.yaml", verbose: bool = True) -> str:
    """전체 파이프라인 실행. run_id 반환."""
    cfg = load_config(config_path)
    W = get_window(cfg)
    assets = get_assets(cfg)
    out_dir = cfg["data"]["feature_store_dir"]
    db = cfg["data"]["meta_db"]

    # 1) 원시 → 수익률 → 지표 → 187 조립(비정규화)
    prices = load_raw(cfg["data"]["raw_dir"])
    logret = log_returns(prices)
    close = prices.loc[logret.index]
    asset_feat, market_feat = compute_features(close, logret, cfg)
    state_all = assemble_state_matrix(logret, asset_feat, market_feat, cfg)

    # 익일 수익률(targets): fwd_ret[t] = logret[t+1] (보상·백테스트용, 정규화 안 함)
    fwd = logret.shift(-1)
    fwd.columns = [f"fwd_ret_{a}" for a in assets]

    # 2) fold 경계
    folds = make_folds(state_all.index, cfg)

    # 3) 메타 초기화
    run_id = f"{fs.config_hash(cfg)}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    created_at = datetime.now().isoformat(timespec="seconds")
    fs.init_meta_db(db)
    fs.write_run(db, run_id, created_at, fs.config_hash(cfg), W, get_state_dim(cfg), assets)
    fs.write_feature_columns(db, run_id, schema.feature_names(W))

    if verbose:
        print(f"run_id={run_id} | state_all {state_all.shape} | folds {len(folds)}")

    # 4) fold별 정규화 fit(train) → 적재
    for fold in folds:
        train_raw = state_all.loc[fold.train_start : fold.train_end]
        scaler = ZScoreScaler.from_config(cfg).fit(train_raw, W)  # ★ train에서만 fit

        split_ranges = {
            "train": (fold.train_start, fold.train_end),
            "valid": (fold.valid_start, fold.valid_end),
            "test": (fold.test_start, fold.test_end),
        }
        rows_report = {}
        for split_name in SPLITS:
            s, e = split_ranges[split_name]
            raw = state_all.loc[s:e]
            norm = scaler.transform(raw)
            fs.write_features(norm, out_dir, fold.fold_id, split_name,
                              expected_columns=schema.feature_names(W))
            # targets: 같은 구간, 익일 없는 마지막 행 drop
            tgt = fwd.reindex(raw.index).dropna(how="any")
            fs.write_targets(tgt, out_dir, fold.fold_id, split_name)
            rows_report[split_name] = (len(norm), len(tgt))

        fs.write_fold(db, run_id, fold.to_meta_dict())
        fs.write_scaler_stats(db, run_id, fold.fold_id, scaler.to_stats_rows())

        if verbose:
            r = rows_report
            print(
                f"  fold{fold.fold_id}: train {r['train'][0]} / valid {r['valid'][0]} / "
                f"test {r['test'][0]} 행 (컬럼 {norm.shape[1]}) | "
                f"test {fold.test_start.date()}~{fold.test_end.date()}"
            )

    if verbose:
        print(f"적재 완료: {out_dir} · 메타 {db}")
    return run_id


def main() -> None:
    build()


if __name__ == "__main__":
    main()
