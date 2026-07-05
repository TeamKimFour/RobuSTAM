"""z-score 정규화 — fit/transform 분리 (룩어헤드 방어의 핵심).

통계량(μ/σ)은 **각 walk-forward fold의 train 구간에서만 fit**하고, valid/test에는 그 통계를
적용만 한다(transform). fit 이후 통계는 immutable — valid/test 데이터가 통계를 바꾸지 못한다.

스코프(config `normalize`):
  - 수익률 150칸(per_asset): 자산당 단일 μ/σ(30칸 풀링) → 30칸 동일 적용.
  - 지표30 + 시장2(per_column): 컬럼별 μ/σ.
  - 직전비중 5칸: 정규화 제외(Gym이 런타임에 채움, 0 유지).
  - std<eps 컬럼(SHV 등 상수): σ=1로 대체해 0-division·inf 방지.
"""

import numpy as np
import pandas as pd

from src.data import schema


class ZScoreScaler:
    def __init__(
        self,
        returns_scope: str = "per_asset",
        feature_scope: str = "per_column",
        eps: float = 1e-8,
    ):
        self.returns_scope = returns_scope
        self.feature_scope = feature_scope
        self.eps = eps
        self.stats_: dict[str, tuple[float, float]] = {}  # col -> (mean, std)

    @classmethod
    def from_config(cls, cfg: dict) -> "ZScoreScaler":
        n = cfg["normalize"]
        return cls(n["returns_scope"], n["feature_scope"], float(n["eps"]))

    def _mean_std(self, values) -> tuple[float, float]:
        arr = np.asarray(values, dtype=float)
        mu = float(np.mean(arr))
        sd = float(np.std(arr))  # 모집단 표준편차(ddof=0)로 일관
        return mu, (sd if sd >= self.eps else 1.0)  # std=0 가드

    def fit(self, train_df: pd.DataFrame, W: int) -> "ZScoreScaler":
        """train 구간에서만 μ/σ를 계산한다. valid/test는 절대 보지 않는다."""
        stats: dict[str, tuple[float, float]] = {}

        # 수익률 블록
        for asset in schema.ASSETS:
            cols = [f"ret_{asset}_lag{k}" for k in range(W)]
            if self.returns_scope == "per_asset":
                mu, sd = self._mean_std(train_df[cols].to_numpy().ravel())
                for c in cols:
                    stats[c] = (mu, sd)
            else:
                for c in cols:
                    stats[c] = self._mean_std(train_df[c])

        # 지표 + 시장 블록: 컬럼별
        for c in train_df.columns:
            if c.startswith("feat_") or c.startswith("mkt_"):
                stats[c] = self._mean_std(train_df[c])

        # prev_weight: 정규화 대상 아님(skip)
        self.stats_ = stats
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """저장된 통계로 정규화만 적용한다(재fit 없음). prev_weight는 그대로."""
        if not self.stats_:
            raise RuntimeError("fit 먼저 호출해야 합니다")
        out = df.copy()
        for c, (mu, sd) in self.stats_.items():
            if c in out.columns:
                out[c] = (out[c] - mu) / sd
        return out

    def to_stats_rows(self) -> list[tuple[str, float, float]]:
        """feature_store.write_scaler_stats 적재용 (feature_name, mean, std)."""
        return [(c, mu, sd) for c, (mu, sd) in self.stats_.items()]

    @classmethod
    def from_stats_rows(
        cls, rows: list[tuple[str, float, float]], **kwargs
    ) -> "ZScoreScaler":
        """저장된 통계로 스케일러 복원(도현 추론 API 재사용)."""
        obj = cls(**kwargs)
        obj.stats_ = {c: (float(mu), float(sd)) for c, mu, sd in rows}
        return obj
