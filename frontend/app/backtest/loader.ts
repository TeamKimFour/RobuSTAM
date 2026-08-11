/**
 * 백테스트 데이터 로더 — 실데이터 JSON이 있으면 그걸, 없으면 mock을 반환한다.
 *
 * 실데이터 경로: `frontend/public/backtest.json` (파이썬 익스포터 `python -m src.backtest.export`가
 * 생성). 파일이 없으면(로컬 개발·모델 미학습 환경) 조용히 mock으로 fallback한다.
 *
 * Server-only: Node의 fs를 쓰므로 Server Component에서만 호출해야 한다.
 * 노출은 3개 위젯만: PerformanceChart · BenchmarkTable · FoldComparisonCard.
 * 나머지(quantstats·기여도·상관·월별 등) mock 위젯은 6주 작업에서 교체.
 */

import { promises as fs } from "node:fs";
import path from "node:path";

import {
  BACKTEST_METRICS,
  BACKTEST_SERIES,
  BENCHMARK_TABLE,
  COMBOS_MOCK,
  FOLD_TABLE,
  type BacktestMetrics,
  type BenchmarkRow,
  type ComboSnapshot,
  type FoldRow,
  type Series,
} from "./mock";

// 백엔드(runner.py)의 전략 키 → FE 표시 이름 매핑. mock과 색이 겹치도록 정렬 순서도 맞춘다.
const DISPLAY_NAME: Record<string, string> = {
  "RL policy": "RobuSTAM (PPO)",
  "60:40": "60:40 (SPY/TLT)",
  "1/N": "1/N Equal Weight",
  "B&H": "Buy & Hold (equal)",
};

type ExportStrategy = {
  name: string;
  color: string;
  nav: { date: string; value: number }[];
  metrics: {
    cagr: number;
    sharpe: number;
    mdd: number;
    vol: number;
    total_return: number;
    avg_turnover: number;
    total_cost: number;
  };
};

type ExportComboFoldRow = {
  fold_id: number;
  period: string;
  sharpe: number;
  cagr: number;
  mdd: number;
  avg_turnover: number;
  total_cost: number;
};

type ExportCombo = {
  combo: string;
  display_name: string;
  color: string;
  nav: { date: string; value: number }[];
  metrics: {
    cagr: number;
    sharpe: number;
    mdd: number;
    vol: number;
    total_return: number;
    avg_turnover: number;
    total_cost: number;
  };
  fold_table: ExportComboFoldRow[];
};

type ExportBundle = {
  generated_at: string;
  initial_nav: number;
  periods: { fold_id: number; start: string; end: string }[];
  strategies: ExportStrategy[];
  fold_table: { fold_id: number; period: string; sharpe: number; cagr: number; mdd: number }[];
  comparison?: unknown;
  combos?: ExportCombo[];
};

export type BacktestSnapshot = {
  source: "real" | "mock";
  generatedAt: string | null;
  series: Series[];
  metrics: BacktestMetrics;
  benchmarkTable: BenchmarkRow[];
  foldTable: FoldRow[];
  /**
   * 콤보별(full·M0·M1·M2·M3) RL policy 성과. 실데이터는 도현이 M0~M3 policy를
   * 학습해 `python -m src.backtest.export --combo full=... --combo M0=...`로 낼 때
   * 채워진다. 미탑재(구 export 산출물)면 mock으로 채운다 — mock도 항상 5종을 낸다.
   */
  combos: ComboSnapshot[];
};

const EXPORT_RELATIVE_PATH = "public/backtest.json";

async function readExport(): Promise<ExportBundle | null> {
  const fullPath = path.join(process.cwd(), EXPORT_RELATIVE_PATH);
  try {
    const raw = await fs.readFile(fullPath, "utf-8");
    const parsed = JSON.parse(raw) as ExportBundle;
    if (!Array.isArray(parsed.strategies) || parsed.strategies.length === 0) return null;
    return parsed;
  } catch {
    return null;
  }
}

function toSeries(strategies: ExportStrategy[]): Series[] {
  return strategies.map((s) => ({
    name: DISPLAY_NAME[s.name] ?? s.name,
    color: s.color,
    points: s.nav.map((p) => ({ date: p.date, value: p.value })),
  }));
}

function toBenchmarkTable(strategies: ExportStrategy[]): BenchmarkRow[] {
  return strategies.map((s) => ({
    name: DISPLAY_NAME[s.name] ?? s.name,
    cagr: s.metrics.cagr,
    sharpe: s.metrics.sharpe,
    mdd: s.metrics.mdd,
    vol: s.metrics.vol,
  }));
}

function toFoldTable(rows: ExportBundle["fold_table"]): FoldRow[] {
  return rows.map((r) => ({
    fold: `Fold ${r.fold_id}`,
    period: r.period,
    sharpe: r.sharpe,
    cagr: r.cagr,
    mdd: r.mdd,
  }));
}

function metricsFromPolicy(strategies: ExportStrategy[]): BacktestMetrics {
  const policy =
    strategies.find((s) => s.name === "RL policy") ?? strategies[0];
  return {
    cagr: policy.metrics.cagr,
    sharpe: policy.metrics.sharpe,
    mdd: policy.metrics.mdd,
    vol: policy.metrics.vol,
  };
}

function toCombos(combos: ExportCombo[] | undefined): ComboSnapshot[] {
  if (!combos || combos.length === 0) return COMBOS_MOCK;
  return combos.map((c) => ({
    combo: c.combo,
    displayName: c.display_name,
    color: c.color,
    points: c.nav.map((p) => ({ date: p.date, value: p.value })),
    metrics: c.metrics,
    foldTable: c.fold_table.map((r) => ({
      fold: `Fold ${r.fold_id}`,
      period: r.period,
      sharpe: r.sharpe,
      cagr: r.cagr,
      mdd: r.mdd,
      avgTurnover: r.avg_turnover,
      totalCost: r.total_cost,
    })),
  }));
}

/** 실데이터 우선 로더. 없으면 mock으로 fallback한다. */
export async function loadBacktestSnapshot(): Promise<BacktestSnapshot> {
  const bundle = await readExport();
  if (!bundle) {
    return {
      source: "mock",
      generatedAt: null,
      series: BACKTEST_SERIES,
      metrics: BACKTEST_METRICS,
      benchmarkTable: BENCHMARK_TABLE,
      foldTable: FOLD_TABLE,
      combos: COMBOS_MOCK,
    };
  }
  return {
    source: "real",
    generatedAt: bundle.generated_at,
    series: toSeries(bundle.strategies),
    metrics: metricsFromPolicy(bundle.strategies),
    benchmarkTable: toBenchmarkTable(bundle.strategies),
    foldTable: toFoldTable(bundle.fold_table),
    combos: toCombos(bundle.combos),
  };
}
