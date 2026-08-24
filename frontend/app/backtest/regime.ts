/**
 * 시장 국면 격자 SSOT — walk-forward test 구간을 국면으로 라벨링하고 필터한다.
 *
 * `docs/data_pipeline.md §4-1` 에 확정된 fold ↔ 국면 매핑을 따른다:
 *   Fold 1 (2020–2021) — COVID 폭락·회복
 *   Fold 2 (2022–2023) — 고금리·채권 급락
 *   Fold 3 (2024–2025) — 최근
 *
 * 민지 국면 라벨 데이터셋(별도 SQLite `market_regimes` 테이블, `docs/db_schema.md`)이
 * 붙기 전까지는 fold 단위가 사실상 국면 격자다. 라벨 데이터가 오면 여기서 REGIMES를
 * 그 라벨로 교체하면 스위처·필터가 그대로 그 세밀도로 동작한다.
 */

export type RegimeId = "all" | "fold1" | "fold2" | "fold3";

export type Regime = {
  id: RegimeId;
  /** 스위처 버튼 라벨 (짧게) */
  shortLabel: string;
  /** 헤더/설명 라벨 (긴 형태) */
  fullLabel: string;
  /** 부가 설명, 배지·툴팁용 */
  description: string;
  /** 이 국면의 시작일 (YYYY-MM-DD). null = 전 구간 */
  start: string | null;
  /** 이 국면의 종료일 (YYYY-MM-DD, 포함). null = 전 구간 */
  end: string | null;
  /** 이 국면에 대응하는 fold_id (없으면 null). FoldComparisonCard 필터에 사용. */
  foldId: number | null;
};

export const REGIMES: Regime[] = [
  {
    id: "all",
    shortLabel: "전 구간",
    fullLabel: "전 구간 (walk-forward 3 fold)",
    description: "3 fold 누적 성과. 국면 필터를 걸지 않는 기본 뷰.",
    start: null,
    end: null,
    foldId: null,
  },
  {
    id: "fold1",
    shortLabel: "COVID",
    fullLabel: "Fold 1 · COVID 폭락·회복 (2020–2021)",
    description: "2020-Q1 급락과 유동성 랠리. 급격한 변동성 확대·자산간 상관 급등 국면.",
    start: "2020-01-01",
    end: "2021-12-31",
    foldId: 1,
  },
  {
    id: "fold2",
    shortLabel: "고금리",
    fullLabel: "Fold 2 · 고금리·채권 급락 (2022–2023)",
    description: "미 연준 급격한 금리인상. 채권(TLT)·성장주 동반 하락, 60:40이 유례없이 실패한 구간.",
    start: "2022-01-01",
    end: "2023-12-31",
    foldId: 2,
  },
  {
    id: "fold3",
    shortLabel: "최근",
    fullLabel: "Fold 3 · 최근 (2024–2025)",
    description: "금리 완화 기대·AI 랠리·지정학 국면 혼재. 학습 데이터 밖의 최신 OOS.",
    start: "2024-01-01",
    end: "2025-12-31",
    foldId: 3,
  },
];

export function resolveRegime(id: string | undefined): Regime {
  return REGIMES.find((r) => r.id === id) ?? REGIMES[0];
}

/**
 * 날짜 문자열(YYYY-MM-DD) 기반 필터. regime.start~end 범위 안에 드는 원소만 반환.
 * regime.start가 null이면(=전 구간) 원본을 그대로 반환한다.
 *
 * 문자열 비교로 판정하므로 ISO 8601 날짜(YYYY-MM-DD) 규약을 가정한다 —
 * 이는 backtest.json 계약(src/backtest/export.py)과 일치한다.
 */
export function filterByRegime<T extends { date: string }>(items: T[], regime: Regime): T[] {
  if (regime.start === null || regime.end === null) return items;
  const start = regime.start;
  const end = regime.end;
  return items.filter((it) => it.date >= start && it.date <= end);
}
