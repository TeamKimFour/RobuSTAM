import { ASSET_ORDER, ASSET_COLOR, type AssetSymbol } from "../lib/assets";

export type SeriesPoint = { date: string; value: number };
export type Series = { name: string; color: string; points: SeriesPoint[] };

export type BacktestMetrics = {
  cagr: number;
  sharpe: number;
  mdd: number;
  vol: number;
};

export type ExtendedRisk = {
  sortino: number;
  calmar: number;
  var95: number;
  cvar95: number;
  skew: number;
  kurtosis: number;
  beta: number;
  bestDay: number;
  worstDay: number;
  hitRatio: number;
};

export type BenchmarkRow = {
  name: string;
  cagr: number;
  sharpe: number;
  mdd: number;
  vol: number;
};

export type MonthlyReturn = { year: number; month: number; ret: number };
export type DayReturn = { date: string; ret: number };
export type FoldRow = { fold: string; period: string; sharpe: number; cagr: number; mdd: number };

const START = new Date("2020-01-01");
const DAYS = 252 * 6;
const STEP = 5;

function mulberry32(seed: number) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function gauss(rand: () => number) {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function buildSeries(name: string, color: string, seed: number, driftAnnual: number, volAnnual: number): Series {
  const rand = mulberry32(seed);
  const dailyDrift = driftAnnual / 252;
  const dailyVol = volAnnual / Math.sqrt(252);
  const points: SeriesPoint[] = [];
  let nav = 1;
  for (let i = 0; i < DAYS; i += 1) {
    const z = gauss(rand);
    nav *= 1 + dailyDrift + dailyVol * z;
    if (i % STEP === 0) {
      const d = new Date(START.getTime());
      d.setDate(d.getDate() + i);
      points.push({ date: d.toISOString().slice(0, 10), value: nav });
    }
  }
  return { name, color, points };
}

// ── 자산별 일별 수익률 시계열 (5×N) ──
function buildAssetReturns(seed: number, driftAnnual: number, volAnnual: number, len: number) {
  const rand = mulberry32(seed);
  const dailyDrift = driftAnnual / 252;
  const dailyVol = volAnnual / Math.sqrt(252);
  const rets: number[] = [];
  for (let i = 0; i < len; i += 1) {
    rets.push(dailyDrift + dailyVol * gauss(rand));
  }
  return rets;
}

const ASSET_PARAMS: Record<AssetSymbol, { seed: number; drift: number; vol: number }> = {
  SPY: { seed: 101, drift: 0.09, vol: 0.18 },
  EWY: { seed: 103, drift: 0.05, vol: 0.22 },
  TLT: { seed: 107, drift: 0.03, vol: 0.14 },
  GLD: { seed: 109, drift: 0.06, vol: 0.16 },
  SHV: { seed: 113, drift: 0.02, vol: 0.01 },
};

export const DAILY_LEN = DAYS;
export const DAILY_DATES: string[] = Array.from({ length: DAYS }, (_, i) => {
  const d = new Date(START.getTime());
  d.setDate(d.getDate() + i);
  return d.toISOString().slice(0, 10);
});

export const ASSET_DAILY_RETURNS: Record<AssetSymbol, number[]> = ASSET_ORDER.reduce(
  (acc, sym) => {
    const p = ASSET_PARAMS[sym];
    acc[sym] = buildAssetReturns(p.seed, p.drift, p.vol, DAYS);
    return acc;
  },
  {} as Record<AssetSymbol, number[]>,
);

// ── 정책 비중 시계열 (mock, N×5) ──
// 시장 국면에 따라 완만하게 변화하는 로직으로 생성 (진짜 정책 아님, 그럴싸한 mock)
function buildWeightSchedule(): Record<AssetSymbol, number[]> {
  const rand = mulberry32(31);
  const base: Record<AssetSymbol, number> = { SPY: 0.35, EWY: 0.15, TLT: 0.2, GLD: 0.2, SHV: 0.1 };
  const out: Record<AssetSymbol, number[]> = ASSET_ORDER.reduce((acc, sym) => {
    acc[sym] = [];
    return acc;
  }, {} as Record<AssetSymbol, number[]>);
  const cur: Record<AssetSymbol, number> = { ...base };
  for (let i = 0; i < DAYS; i += 1) {
    // 저주파 랜덤워크 + 재정규화
    for (const sym of ASSET_ORDER) {
      cur[sym] = Math.max(0.01, cur[sym] + (rand() - 0.5) * 0.02);
    }
    const total = ASSET_ORDER.reduce((s, k) => s + cur[k], 0);
    for (const sym of ASSET_ORDER) {
      cur[sym] /= total;
      out[sym].push(cur[sym]);
    }
  }
  return out;
}

export const WEIGHT_SCHEDULE = buildWeightSchedule();

// ── 정책 NAV: 자산수익 · 비중 · 수수료로 재구성 (turnover 계산용) ──
const TRANSACTION_COST = 0.001;

function buildPortfolioSeries() {
  const nav: number[] = [1];
  const turnover: number[] = [0];
  const cost: number[] = [0];
  const dailyRet: number[] = [0];
  const contribution: Record<AssetSymbol, number[]> = ASSET_ORDER.reduce((acc, s) => {
    acc[s] = [0];
    return acc;
  }, {} as Record<AssetSymbol, number[]>);

  for (let i = 1; i < DAYS; i += 1) {
    let to = 0;
    for (const sym of ASSET_ORDER) {
      to += Math.abs(WEIGHT_SCHEDULE[sym][i] - WEIGHT_SCHEDULE[sym][i - 1]);
    }
    turnover.push(to);
    const feeRate = to * TRANSACTION_COST;
    cost.push(nav[i - 1] * feeRate);

    let ret = 0;
    for (const sym of ASSET_ORDER) {
      const c = WEIGHT_SCHEDULE[sym][i] * ASSET_DAILY_RETURNS[sym][i];
      ret += c;
      contribution[sym].push(c);
    }
    const netRet = ret - feeRate;
    dailyRet.push(netRet);
    nav.push(nav[i - 1] * (1 + netRet));
  }
  return { nav, turnover, cost, dailyRet, contribution };
}

const PORT = buildPortfolioSeries();

const RL_POINTS = DAILY_DATES.filter((_, i) => i % STEP === 0).map((date, k) => ({
  date,
  value: PORT.nav[k * STEP],
}));
const RL: Series = { name: "RobuSTAM (PPO)", color: "#8b5cf6", points: RL_POINTS };
const EQ_60_40 = buildSeries("60:40 (SPY/TLT)", "#10b981", 7, 0.07, 0.11);
const EQ_WEIGHT = buildSeries("1/N Equal Weight", "#a5b4fc", 11, 0.06, 0.10);
const BUY_HOLD_SPY = buildSeries("Buy & Hold SPY", "#dc2626", 23, 0.09, 0.18);

export const BACKTEST_SERIES: Series[] = [RL, EQ_60_40, EQ_WEIGHT, BUY_HOLD_SPY];

function drawdownFromPoints(points: SeriesPoint[]): number {
  let peak = points[0].value;
  let mdd = 0;
  for (const p of points) {
    if (p.value > peak) peak = p.value;
    const dd = p.value / peak - 1;
    if (dd < mdd) mdd = dd;
  }
  return mdd;
}

function cagrFromPoints(points: SeriesPoint[]): number {
  const first = points[0].value;
  const last = points[points.length - 1].value;
  const years = (points.length * STEP) / 252;
  return Math.pow(last / first, 1 / years) - 1;
}

function pointReturns(points: SeriesPoint[]): number[] {
  const rets: number[] = [];
  for (let i = 1; i < points.length; i += 1) {
    rets.push(points[i].value / points[i - 1].value - 1);
  }
  return rets;
}

function volFromPoints(points: SeriesPoint[]): number {
  const rets = pointReturns(points);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / rets.length;
  return Math.sqrt(variance) * Math.sqrt(252 / STEP);
}

function sharpeFromPoints(points: SeriesPoint[]): number {
  const rets = pointReturns(points);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / rets.length;
  const std = Math.sqrt(variance);
  if (std === 0) return 0;
  return (mean / std) * Math.sqrt(252 / STEP);
}

export const BACKTEST_METRICS: BacktestMetrics = {
  cagr: cagrFromPoints(RL.points),
  sharpe: sharpeFromPoints(RL.points),
  mdd: drawdownFromPoints(RL.points),
  vol: volFromPoints(RL.points),
};

/**
 * 임의의 NAV 시계열에서 KPI 4종을 재산출한다. 국면 필터로 잘린 구간에서도 지표를
 * 다시 뽑기 위해 공개한다. 표본이 부족(<2)하면 0으로 채워 UI가 깨지지 않게 한다.
 */
export function metricsFromPoints(points: SeriesPoint[]): BacktestMetrics {
  if (points.length < 2) return { cagr: 0, sharpe: 0, mdd: 0, vol: 0 };
  return {
    cagr: cagrFromPoints(points),
    sharpe: sharpeFromPoints(points),
    mdd: drawdownFromPoints(points),
    vol: volFromPoints(points),
  };
}

export const BENCHMARK_TABLE: BenchmarkRow[] = BACKTEST_SERIES.map((s) => ({
  name: s.name,
  cagr: cagrFromPoints(s.points),
  sharpe: sharpeFromPoints(s.points),
  mdd: drawdownFromPoints(s.points),
  vol: volFromPoints(s.points),
}));

export function drawdownSeries(points: SeriesPoint[]): SeriesPoint[] {
  let peak = points[0].value;
  return points.map((p) => {
    if (p.value > peak) peak = p.value;
    return { date: p.date, value: p.value / peak - 1 };
  });
}

// ── 롤링 지표 (252 거래일 창 = 50 샘플[STEP=5]) ──
const ROLL_WINDOW = 50;

export function rollingSharpe(points: SeriesPoint[]): SeriesPoint[] {
  const rets = pointReturns(points);
  const out: SeriesPoint[] = [];
  for (let i = ROLL_WINDOW; i < rets.length; i += 1) {
    const slice = rets.slice(i - ROLL_WINDOW, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((s, r) => s + (r - mean) ** 2, 0) / slice.length;
    const std = Math.sqrt(variance);
    const sharpe = std === 0 ? 0 : (mean / std) * Math.sqrt(252 / STEP);
    out.push({ date: points[i + 1].date, value: sharpe });
  }
  return out;
}

export function rollingVol(points: SeriesPoint[]): SeriesPoint[] {
  const rets = pointReturns(points);
  const out: SeriesPoint[] = [];
  for (let i = ROLL_WINDOW; i < rets.length; i += 1) {
    const slice = rets.slice(i - ROLL_WINDOW, i);
    const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
    const variance = slice.reduce((s, r) => s + (r - mean) ** 2, 0) / slice.length;
    out.push({ date: points[i + 1].date, value: Math.sqrt(variance) * Math.sqrt(252 / STEP) });
  }
  return out;
}

// ── 월별 수익률 히트맵 (RL 기준) ──
export function monthlyReturns(points: SeriesPoint[]): MonthlyReturn[] {
  const map = new Map<string, { start: number; end: number }>();
  for (const p of points) {
    const key = p.date.slice(0, 7);
    const cur = map.get(key);
    if (!cur) map.set(key, { start: p.value, end: p.value });
    else cur.end = p.value;
  }
  const out: MonthlyReturn[] = [];
  for (const [key, v] of map) {
    const [y, m] = key.split("-").map(Number);
    out.push({ year: y, month: m, ret: v.end / v.start - 1 });
  }
  return out.sort((a, b) => (a.year - b.year) * 12 + (a.month - b.month));
}

// ── Turnover / 거래비용 시계열 (자산비중 스케줄 기반) ──
export const TURNOVER_SERIES: SeriesPoint[] = DAILY_DATES.filter((_, i) => i % STEP === 0).map(
  (date, k) => ({ date, value: PORT.turnover[k * STEP] }),
);

export const CUM_COST_SERIES: SeriesPoint[] = (() => {
  let cum = 0;
  const out: SeriesPoint[] = [];
  for (let i = 0; i < DAYS; i += 1) {
    cum += PORT.cost[i];
    if (i % STEP === 0) out.push({ date: DAILY_DATES[i], value: cum });
  }
  return out;
})();

// ── 자산별 누적 P&L 기여도 (스택 영역차트용) ──
export const ASSET_CONTRIBUTION_SERIES: Series[] = ASSET_ORDER.map((sym) => {
  let cum = 0;
  const pts: SeriesPoint[] = [];
  for (let i = 0; i < DAYS; i += 1) {
    cum += PORT.contribution[sym][i];
    if (i % STEP === 0) pts.push({ date: DAILY_DATES[i], value: cum });
  }
  return { name: sym, color: ASSET_COLOR[sym], points: pts };
});

// ── Alpha (RL - benchmark) 시계열 ──
export function alphaSeries(strategy: Series, benchmark: Series): SeriesPoint[] {
  const map = new Map(benchmark.points.map((p) => [p.date, p.value]));
  const out: SeriesPoint[] = [];
  for (const p of strategy.points) {
    const b = map.get(p.date);
    if (b == null) continue;
    out.push({ date: p.date, value: p.value / b - 1 });
  }
  return out;
}

// ── Fold별 walk-forward 성과 (mock) ──
export const FOLD_TABLE: FoldRow[] = [
  { fold: "Fold 1", period: "2020-01 ~ 2021-12", sharpe: 0.88, cagr: 0.12, mdd: -0.18 },
  { fold: "Fold 2", period: "2022-01 ~ 2023-12", sharpe: 1.05, cagr: 0.09, mdd: -0.11 },
  { fold: "Fold 3", period: "2024-01 ~ 2025-12", sharpe: 0.72, cagr: 0.08, mdd: -0.15 },
];

// ── Best / Worst 거래일 (일간 수익률 상위/하위) ──
// start/end(YYYY-MM-DD)를 넘기면 그 구간으로 한정한다 (국면 필터). 생략 시 전 구간.
export function bestWorstDays(
  n: number,
  start?: string | null,
  end?: string | null,
): { best: DayReturn[]; worst: DayReturn[] } {
  const withDate: DayReturn[] = PORT.dailyRet
    .map((r, i) => ({ date: DAILY_DATES[i], ret: r }))
    .filter((d) => (!start || d.date >= start) && (!end || d.date <= end));
  const sorted = [...withDate].sort((a, b) => b.ret - a.ret);
  return { best: sorted.slice(0, n), worst: sorted.slice(-n).reverse() };
}

// ── 확장 리스크 지표 ──
export const EXTENDED_RISK: ExtendedRisk = (() => {
  const rets = PORT.dailyRet.slice(1);
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / rets.length;
  const std = Math.sqrt(variance);
  const downside = rets.filter((r) => r < 0);
  const downMean =
    downside.reduce((s, r) => s + (r - 0) ** 2, 0) / (downside.length || 1);
  const sortino = std === 0 ? 0 : (mean / Math.sqrt(downMean)) * Math.sqrt(252);
  const cagr = BACKTEST_METRICS.cagr;
  const mdd = BACKTEST_METRICS.mdd;
  const calmar = mdd === 0 ? 0 : cagr / Math.abs(mdd);

  const sorted = [...rets].sort((a, b) => a - b);
  const var95 = sorted[Math.floor(sorted.length * 0.05)];
  const tail = sorted.slice(0, Math.floor(sorted.length * 0.05) + 1);
  const cvar95 = tail.reduce((a, b) => a + b, 0) / tail.length;

  const m3 = rets.reduce((s, r) => s + (r - mean) ** 3, 0) / rets.length;
  const m4 = rets.reduce((s, r) => s + (r - mean) ** 4, 0) / rets.length;
  const skew = m3 / Math.pow(std, 3);
  const kurtosis = m4 / Math.pow(std, 4) - 3;

  const spyRets = ASSET_DAILY_RETURNS.SPY.slice(1);
  const covXY =
    rets.reduce((s, r, i) => s + (r - mean) * (spyRets[i] - meanOf(spyRets)), 0) / rets.length;
  const beta = covXY / variance;

  const hitRatio = rets.filter((r) => r > 0).length / rets.length;
  const bestDay = Math.max(...rets);
  const worstDay = Math.min(...rets);

  return { sortino, calmar, var95, cvar95, skew, kurtosis, beta, bestDay, worstDay, hitRatio };
})();

function meanOf(arr: number[]): number {
  return arr.reduce((a, b) => a + b, 0) / arr.length;
}

// ── 상관계수 매트릭스 (자산 5×5) ──
export const CORRELATION_MATRIX: number[][] = (() => {
  const arrs = ASSET_ORDER.map((s) => ASSET_DAILY_RETURNS[s]);
  const means = arrs.map(meanOf);
  const stds = arrs.map((a, i) => {
    const m = means[i];
    return Math.sqrt(a.reduce((s, r) => s + (r - m) ** 2, 0) / a.length);
  });
  const n = arrs[0].length;
  return arrs.map((a, i) =>
    arrs.map((b, j) => {
      let cov = 0;
      for (let k = 0; k < n; k += 1) cov += (a[k] - means[i]) * (b[k] - means[j]);
      cov /= n;
      const denom = stds[i] * stds[j];
      return denom === 0 ? 0 : cov / denom;
    }),
  );
})();

// ── Underwater 지속기간 (Drawdown 연속 일수) ──
export function underwaterDurations(points: SeriesPoint[]): number[] {
  let peak = points[0].value;
  let run = 0;
  const runs: number[] = [];
  for (const p of points) {
    if (p.value >= peak) {
      if (run > 0) runs.push(run * STEP);
      run = 0;
      peak = p.value;
    } else {
      run += 1;
    }
  }
  if (run > 0) runs.push(run * STEP);
  return runs;
}

// ── 시장 상태 (Inference 페이지용): 자산별 지표 mock + 시장공통 게이지 ──
export type AssetIndicators = {
  symbol: AssetSymbol;
  rsi: number;
  macdHist: number;
  roc10: number;
  vol20: number;
  bbWidth: number;
  maCross: "golden" | "dead" | "neutral";
};

export const ASSET_INDICATORS: AssetIndicators[] = [
  { symbol: "SPY", rsi: 58.2, macdHist: 0.42, roc10: 2.1, vol20: 0.13, bbWidth: 0.045, maCross: "golden" },
  { symbol: "EWY", rsi: 47.6, macdHist: -0.08, roc10: 0.4, vol20: 0.18, bbWidth: 0.052, maCross: "neutral" },
  { symbol: "TLT", rsi: 39.1, macdHist: -0.35, roc10: -1.8, vol20: 0.11, bbWidth: 0.038, maCross: "dead" },
  { symbol: "GLD", rsi: 55.0, macdHist: 0.12, roc10: 1.0, vol20: 0.14, bbWidth: 0.041, maCross: "golden" },
  { symbol: "SHV", rsi: 51.2, macdHist: 0.01, roc10: 0.05, vol20: 0.005, bbWidth: 0.002, maCross: "neutral" },
];

export const MARKET_GAUGES = {
  equityBondRatio: 1.42,          // SPY/TLT 비율의 MA20 (>1 = 주식 우위)
  goldVolRatio: 0.82,             // 금 단기/장기 변동성 비율 (>1 = 금 리스크 확대)
};

// ── 자산 30일 로그수익률 히트맵 (5×30, 인퍼런스 페이지) ──
export const RECENT_RETURNS_30D: Record<AssetSymbol, number[]> = ASSET_ORDER.reduce((acc, sym) => {
  acc[sym] = ASSET_DAILY_RETURNS[sym].slice(-30);
  return acc;
}, {} as Record<AssetSymbol, number[]>);

export const RECENT_RETURNS_DATES: string[] = DAILY_DATES.slice(-30);

// ── 국면(=fold)별 RL vs 벤치마크 비교 (8주차 발표 하이라이트) ───────
// backtest.json 계약(`comparison[].vs_benchmark`)의 FE 소비 형태.
// CLAUDE.md §1 판정: 샤프 15%+ 개선 또는 MDD 20%+ 방어면 beatsTarget=true.
// mock 값은 노션 국면 매핑(Fold 1=COVID / 2=고금리 / 3=최근)에 실감을 준 것 —
// 도현·찬휘 실 산출물이 오면 export.py가 이 값을 덮어쓴다.
//
// 각 국면 시나리오:
//   Fold 1(COVID): 폭락 방어에서 RL이 60:40 대비 강함, 1/N도 이김, B&H는 못 이김
//   Fold 2(고금리): 채권 급락으로 60:40 무너져 RL이 크게 이김, 1/N은 근소, B&H는 소폭 이김
//   Fold 3(최근):  SPY 랠리에서 B&H가 강해 RL이 소폭 밀림, 60:40은 이김, 1/N 근소

export type BenchmarkComparisonEntry = {
  benchmark: string;              // FE 표시명 (loader.ts DISPLAY_NAME 매핑 후)
  sharpeImprovementPct: number;   // (RL_sharpe - bench_sharpe) / |bench_sharpe|
  mddDefensePct: number;          // (|bench_mdd| - |RL_mdd|) / |bench_mdd|
  beatsTarget: boolean;
};

export type FoldBenchmarkComparison = {
  foldId: number;
  entries: BenchmarkComparisonEntry[];
};

export const COMPARISON_MOCK: FoldBenchmarkComparison[] = [
  {
    foldId: 1,
    entries: [
      { benchmark: "1/N Equal Weight", sharpeImprovementPct: 0.28, mddDefensePct: 0.18, beatsTarget: true },
      { benchmark: "60:40 (SPY/TLT)",  sharpeImprovementPct: 0.34, mddDefensePct: 0.22, beatsTarget: true },
      { benchmark: "Buy & Hold (equal)", sharpeImprovementPct: -0.08, mddDefensePct: 0.31, beatsTarget: true },
    ],
  },
  {
    foldId: 2,
    entries: [
      { benchmark: "1/N Equal Weight", sharpeImprovementPct: 0.14, mddDefensePct: 0.11, beatsTarget: false },
      { benchmark: "60:40 (SPY/TLT)",  sharpeImprovementPct: 0.62, mddDefensePct: 0.38, beatsTarget: true },
      { benchmark: "Buy & Hold (equal)", sharpeImprovementPct: 0.09, mddDefensePct: 0.24, beatsTarget: true },
    ],
  },
  {
    foldId: 3,
    entries: [
      { benchmark: "1/N Equal Weight", sharpeImprovementPct: 0.02, mddDefensePct: 0.06, beatsTarget: false },
      { benchmark: "60:40 (SPY/TLT)",  sharpeImprovementPct: 0.19, mddDefensePct: 0.12, beatsTarget: true },
      { benchmark: "Buy & Hold (equal)", sharpeImprovementPct: -0.15, mddDefensePct: 0.04, beatsTarget: false },
    ],
  },
];
