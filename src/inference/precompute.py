"""precompute — 오늘의 State → 익일 추천 비중(latest.json) 생산자.

학습 끝단(policy.zip)과 API 시작단(latest.json 소비)을 잇는 다리. 민지(데이터 글루)와
도현(모델 predict)의 접점이며, 계약은 민지가 확정한다.

접점 계약 (변경 금지 — 도현 통보):
    build_today_obs(cfg, prev_weights=None) -> (obs: np.ndarray[state_dim], date: str)
        · obs는 z-score 정규화 완료(config inference.scaler_run_id/fold_id 통계 재현).
        · prev_weight 5칸은 raw 비중(정규화 안 함), None이면 SHV 100% 콜드스타트.
        · 자산순서 [SPY,EWY,TLT,GLD,SHV] 고정.  ← 민지 책임
    generate_latest(cfg, model=None) -> dict
        · obs 받아 predict → softmax → latest.json 원자적 기록.  ← 도현 책임(모델 파트)

전제: `python -m src.data.collect`(원시 캐시) + `python -m src.data.build`(메타DB scaler_stats)가
선행돼야 하고, config `inference` 섹션이 채워져 있어야 한다.

실행: `python -m src.inference.precompute`
"""

from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config_loader import (
    get_inference,
    get_n_asset_features,
    get_n_market_features,
    get_precompute_path,
    get_window,
    load_config,
)
from src.data import schema
from src.data.assemble import assemble_state_matrix
from src.data.collect import load_raw
from src.data.feature_store import (
    config_hash,
    latest_run_id_for_config,
    read_scaler_stats,
)
from src.data.features import compute_features
from src.data.normalize import ZScoreScaler
from src.data.returns import log_returns


# ────────────────────────────── 민지: 데이터 글루 ──────────────────────────────
def _restore_scaler(cfg: dict) -> ZScoreScaler:
    """config inference가 가리키는 build run/fold의 정규화 통계로 스케일러를 복원한다.

    build.py가 fold train에서 fit해 저장한 통계를 그대로 재사용 → 학습과 동일한 z-score를
    룩어헤드 없이 재현한다.

    `scaler_run_id`가 비어 있으면 **현재 config의 `config_hash`로 만들어진 최신 build run**을
    자동 선택한다. build run_id는 타임스탬프를 포함해 머신 간 재현이 불가능하지만, 같은
    config로 재빌드하면 통계가 사실상 동일하므로(실측 최대 상대오차 7.4e-4) CI·서버가 각자
    build한 뒤 자기 run을 쓰면 된다 — `meta.sqlite`를 옮길 필요가 없다(이슈 #33).
    정확한 감사 추적이 필요하면 `scaler_run_id`를 명시해 고정한다.
    """
    inf = get_inference(cfg)
    fold_id = inf.get("scaler_fold_id")
    meta_db = cfg["data"]["meta_db"]

    run_id = inf.get("scaler_run_id")
    if not run_id:
        cfg_hash = config_hash(cfg)
        run_id = latest_run_id_for_config(meta_db, cfg_hash)
        if not run_id:
            raise ValueError(
                f"config_hash={cfg_hash!r}로 만들어진 build run이 없습니다 "
                f"(meta_db={meta_db!r}). `python -m src.data.build`를 먼저 실행하거나, "
                "config inference.scaler_run_id에 사용할 run_id를 명시하세요."
            )
        print(f"[precompute] scaler_run_id 미지정 → 자동 선택: {run_id} (config_hash={cfg_hash})")

    stats = read_scaler_stats(meta_db, run_id, int(fold_id))
    if not stats:
        raise ValueError(
            f"scaler_stats가 비어 있습니다: run_id={run_id!r}, fold_id={fold_id} "
            "(python -m src.data.build 가 먼저 실행돼야 합니다)."
        )
    # read_scaler_stats는 {name:(mean,std)} dict → from_stats_rows는 [(name,mean,std)] 기대.
    rows = [(name, mu, sd) for name, (mu, sd) in stats.items()]
    return ZScoreScaler.from_stats_rows(rows)


def _cold_start_weights() -> np.ndarray:
    """직전 비중 미상 시 콜드스타트 = SHV(현금) 100% (env.reset과 동일)."""
    w = np.zeros(schema.N_ASSETS, dtype=float)
    w[schema.ASSETS.index("SHV")] = 1.0
    return w


def build_today_obs(
    cfg: dict, prev_weights: np.ndarray | None = None
) -> tuple[np.ndarray, str]:
    """오늘(최신 거래일)의 정규화된 관측 1행 + 날짜를 반환한다. (민지 책임)

    build.py와 동일한 순서(수집 캐시 → 로그수익률 → 지표 → 187 조립)로 계산해 **마지막 행**을
    취하고, config inference가 가리키는 fold의 통계로 z-score를 재현한다. 전체 원시로 계산해
    마지막 행이 build 산출물과 정확히 일치한다(정규화 재현 대조 가능). prev_weight 5칸은
    prev_weights(정규화 안 함)로 채우며, None이면 SHV 100% 콜드스타트다.

    Returns
    -------
    obs  : np.ndarray, shape (state_dim,)=187 @ W=30, dtype float32. 컬럼순서=feature_names(W).
    date : str "YYYY-MM-DD" — obs가 대표하는 최신 거래일.
    """
    W = get_window(cfg)

    # 1) build.py와 동일 체인 — 전체 원시로 계산(마지막 행이 build 결과와 정확히 일치).
    prices = load_raw(cfg["data"]["raw_dir"])
    logret = log_returns(prices)
    close = prices.loc[logret.index]
    asset_feat, market_feat = compute_features(close, logret, cfg)
    state_all = assemble_state_matrix(logret, asset_feat, market_feat, cfg)
    if state_all.empty:
        raise ValueError("조립 가능한 State가 없습니다(원시 데이터·warm-up 확인).")

    raw_today = state_all.iloc[[-1]]  # 1행 DataFrame(오늘)
    date = str(raw_today.index[-1].date())

    # 2) 정규화 재현(학습과 동일 통계). prev_weight는 stats에 없어 그대로 통과(0).
    scaler = _restore_scaler(cfg)
    obs = scaler.transform(raw_today).iloc[0].to_numpy(dtype=np.float32)

    # 3) prev_weight 주입 — raw 비중(정규화 안 함). None이면 SHV 100%.
    #    지표 개수가 config로 가변이므로 K_a·K_m을 함께 넘겨 슬라이스를 계산한다.
    pw = np.asarray(prev_weights, dtype=float) if prev_weights is not None else _cold_start_weights()
    if pw.shape != (schema.N_ASSETS,):
        raise ValueError(f"prev_weights는 shape ({schema.N_ASSETS},)여야 합니다: {pw.shape}")
    pw_slice = schema.prev_weight_slice(
        W,
        n_asset_features=get_n_asset_features(cfg),
        n_market_features=get_n_market_features(cfg),
    )
    obs[pw_slice] = pw.astype(np.float32)
    return obs, date


# ────────────────────────────── 도현: 모델 파트 ──────────────────────────────
def _read_prev_weights(cfg: dict) -> np.ndarray | None:
    """직전 latest.json에서 비중을 자산순서로 읽는다. 없거나 불완전하면 None(콜드스타트)."""
    path = Path(get_precompute_path(cfg))
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            weights = json.load(f).get("weights", {})
        return np.array([weights[a] for a in schema.ASSETS], dtype=float)
    except (json.JSONDecodeError, KeyError, TypeError):
        return None


def _atomic_write_json(path: Path, payload: dict) -> None:
    """tmp 파일에 쓰고 rename — 부분쓰기 상태의 파일을 API가 읽는 것을 방지한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _load_policy(model_path: str):
    """SB3 PPO 정책 로드 — 무거운 import는 여기서만. (도현 담당 파트)"""
    from stable_baselines3 import PPO

    return PPO.load(model_path)


def generate_latest(cfg: dict, model=None, now: str | None = None) -> dict:
    """오늘의 익일 추천 비중을 계산해 latest.json으로 쓴다. (도현 책임 — 모델 파트)

    민지의 build_today_obs로 obs를 받아, policy predict → softmax(학습과 동일) →
    LatestInference 계약으로 원자적 기록한다. model 미지정 시 config inference.model_path에서
    PPO를 로드한다(테스트는 목 모델 주입).
    """
    prev = _read_prev_weights(cfg)
    obs, date = build_today_obs(cfg, prev)

    inf = get_inference(cfg)
    if model is None:
        model = _load_policy(inf["model_path"])
    action = np.asarray(model.predict(obs, deterministic=True)[0], dtype=float)

    from src.env.portfolio_env import _softmax  # 학습 step()과 동일 softmax 재사용

    weights = _softmax(action)
    payload = {
        "date": date,
        "generated_at": now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_version": str(inf.get("model_version", "unknown")),
        "weights": {a: float(w) for a, w in zip(schema.ASSETS, weights)},
    }
    _atomic_write_json(Path(get_precompute_path(cfg)), payload)
    return payload


def main() -> None:
    """CLI — `python -m src.inference.precompute`."""
    cfg = load_config()
    payload = generate_latest(cfg)
    print(f"latest.json 생성: {get_precompute_path(cfg)}")
    print(payload)


if __name__ == "__main__":
    main()
