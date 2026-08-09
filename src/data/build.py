"""Feature Store 빌드 오케스트레이터.

수집(캐시) → 로그수익률 → 지표 → 187 조립 → fold별 (train에서 정규화 fit → 전체 transform) →
Parquet 적재 + 익일수익률(targets) 저장 + 메타(folds·scaler_stats) 기록.

지표는 fold와 무관하게 1회만 계산(인과적)하고, **정규화만 fold별로 train에서 재fit**한다.
실행: `python -m src.data.build`
"""

from datetime import datetime
from pathlib import Path

import numpy as np

from src.config_loader import (
    get_asset_features,
    get_assets,
    get_feature_store_dir,
    get_market_features,
    get_meta_db,
    get_state_dim,
    get_window,
    load_config,
    resolve_combo,
)
from src.data import feature_store as fs
from src.data import schema
from src.data.assemble import assemble_state_matrix
from src.data.collect import load_raw
from src.data.features import compute_features
from src.data.normalize import ZScoreScaler
from src.data.returns import log_returns
from src.data.splits import make_folds

SPLITS = ("train", "valid", "test")


def build(config_path: str = "config/config.yaml", combo: str | None = None, verbose: bool = True) -> str:
    """전체 파이프라인 실행. run_id 반환.

    combo 생략 시 config의 active_combo(기본 'full', 기존 187차원)를 빌드해 기존 플랫 경로
    (data/feature_store/fold=*/...)에 그대로 적재한다 — daily.yml·train.py 무영향.
    M0~M3 등 다른 콤보는 combo="M0" 처럼 명시해 별도 하위 디렉토리(data/feature_store/M0/...)에
    병행 빌드한다(docs/data_pipeline.md §3-2).
    """
    cfg = load_config(config_path)
    combo_name = combo if combo is not None else cfg.get("active_combo")
    resolved_cfg = resolve_combo(cfg, combo_name)
    W = get_window(resolved_cfg)
    assets = get_assets(resolved_cfg)
    out_dir = get_feature_store_dir(cfg, combo_name)
    db = get_meta_db(cfg, combo_name)
    names = schema.feature_names(
        W,
        asset_features=get_asset_features(resolved_cfg),
        market_features=get_market_features(resolved_cfg),
    )

    # 1) 원시 → 수익률 → 지표(콤보 반영) → State 조립(비정규화)
    prices = load_raw(cfg["data"]["raw_dir"])
    logret = log_returns(prices)
    close = prices.loc[logret.index]
    asset_feat, market_feat = compute_features(close, logret, resolved_cfg)
    state_all = assemble_state_matrix(logret, asset_feat, market_feat, resolved_cfg)

    # 익일 수익률(targets): fwd_ret[t] = logret[t+1] (보상·백테스트용, 정규화 안 함)
    fwd = logret.shift(-1)
    fwd.columns = [f"fwd_ret_{a}" for a in assets]

    # 2) fold 경계
    folds = make_folds(state_all.index, resolved_cfg)

    # 3) 메타 초기화 — config_hash는 resolved_cfg 기준(콤보별 실제 지표 리스트가 해시에 반영됨)
    config_hash = fs.config_hash(resolved_cfg)
    run_id = f"{config_hash}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    created_at = datetime.now().isoformat(timespec="seconds")
    fs.init_meta_db(db)
    fs.write_run(db, run_id, created_at, config_hash, W, get_state_dim(resolved_cfg), assets,
                 combo=combo_name)
    fs.write_feature_columns(db, run_id, names)

    if verbose:
        print(f"combo={combo_name} | run_id={run_id} | state_all {state_all.shape} | folds {len(folds)}")

    # 4) fold별 정규화 fit(train) → 적재
    for fold in folds:
        train_raw = state_all.loc[fold.train_start : fold.train_end]
        scaler = ZScoreScaler.from_config(resolved_cfg).fit(train_raw, W)  # ★ train에서만 fit

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
            fs.write_features(norm, out_dir, fold.fold_id, split_name, expected_columns=names)
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
