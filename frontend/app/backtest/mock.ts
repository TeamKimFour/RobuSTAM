export type SeriesPoint = { date: string; value: number };
export type Series = { name: string; color: string; points: SeriesPoint[] };

export type BacktestMetrics = {
  cagr: number;
  sharpe: number;
  mdd: number;
  vol: number;
};

export type BenchmarkRow = {
  name: string;
  cagr: number;
  sharpe: number;
  mdd: number;
  vol: number;
};

const START = new Date("2020-01-01");
const DAYS = 252 * 6;

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

function buildSeries(name: string, color: string, seed: number, driftAnnual: number, volAnnual: number): Series {
  const rand = mulberry32(seed);
  const dailyDrift = driftAnnual / 252;
  const dailyVol = volAnnual / Math.sqrt(252);
  const points: SeriesPoint[] = [];
  let nav = 1;
  for (let i = 0; i < DAYS; i += 1) {
    const z = gauss(rand);
    nav *= 1 + dailyDrift + dailyVol * z;
    if (i % 5 === 0) {
      const d = new Date(START.getTime());
      d.setDate(d.getDate() + i);
      points.push({ date: d.toISOString().slice(0, 10), value: nav });
    }
  }
  return { name, color, points };
}

function gauss(rand: () => number) {
  const u1 = Math.max(rand(), 1e-9);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

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
  const years = (points.length * 5) / 252;
  return Math.pow(last / first, 1 / years) - 1;
}

function volFromPoints(points: SeriesPoint[]): number {
  const rets: number[] = [];
  for (let i = 1; i < points.length; i += 1) {
    rets.push(points[i].value / points[i - 1].value - 1);
  }
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / rets.length;
  return Math.sqrt(variance) * Math.sqrt(252 / 5);
}

function sharpeFromPoints(points: SeriesPoint[]): number {
  const rets: number[] = [];
  for (let i = 1; i < points.length; i += 1) {
    rets.push(points[i].value / points[i - 1].value - 1);
  }
  const mean = rets.reduce((a, b) => a + b, 0) / rets.length;
  const variance = rets.reduce((s, r) => s + (r - mean) ** 2, 0) / rets.length;
  const std = Math.sqrt(variance);
  if (std === 0) return 0;
  return (mean / std) * Math.sqrt(252 / 5);
}

const RL = buildSeries("RobuSTAM (PPO)", "#8b5cf6", 42, 0.11, 0.13);
const EQ_60_40 = buildSeries("60:40 (SPY/TLT)", "#10b981", 7, 0.07, 0.11);
const EQ_WEIGHT = buildSeries("1/N Equal Weight", "#a5b4fc", 11, 0.06, 0.10);
const BUY_HOLD_SPY = buildSeries("Buy & Hold SPY", "#dc2626", 23, 0.09, 0.18);

export const BACKTEST_SERIES: Series[] = [RL, EQ_60_40, EQ_WEIGHT, BUY_HOLD_SPY];

export const BACKTEST_METRICS: BacktestMetrics = {
  cagr: cagrFromPoints(RL.points),
  sharpe: sharpeFromPoints(RL.points),
  mdd: drawdownFromPoints(RL.points),
  vol: volFromPoints(RL.points),
};

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
