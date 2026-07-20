import type { ExtendedRisk } from "./mock";

type Props = { risk: ExtendedRisk };

export default function ExtendedRiskCard({ risk }: Props) {
  const items = [
    { label: "Sortino", value: risk.sortino.toFixed(2), hint: "하방 변동성 조정 수익", tone: risk.sortino >= 1 ? "positive" : "muted" },
    { label: "Calmar", value: risk.calmar.toFixed(2), hint: "CAGR / |MDD|", tone: risk.calmar >= 0.5 ? "positive" : "muted" },
    { label: "VaR 95%", value: pct(risk.var95), hint: "일간 손실 하한 5% 분위", tone: "negative" },
    { label: "CVaR 95%", value: pct(risk.cvar95), hint: "테일 손실 기대", tone: "negative" },
    { label: "Skewness", value: risk.skew.toFixed(2), hint: "> 0 이면 오른쪽 꼬리", tone: risk.skew >= 0 ? "positive" : "negative" },
    { label: "Kurtosis (초과)", value: risk.kurtosis.toFixed(2), hint: "> 0 이면 두꺼운 꼬리", tone: "muted" },
    { label: "Beta vs SPY", value: risk.beta.toFixed(2), hint: "시장 민감도", tone: "muted" },
    { label: "Hit Ratio", value: `${(risk.hitRatio * 100).toFixed(1)}%`, hint: "수익 낸 거래일 비율", tone: risk.hitRatio >= 0.5 ? "positive" : "muted" },
    { label: "Best Day", value: pct(risk.bestDay), hint: "역대 최고 일간 수익", tone: "positive" },
    { label: "Worst Day", value: pct(risk.worstDay), hint: "역대 최악 일간 손실", tone: "negative" },
  ] as const;

  return (
    <section className="card">
      <h2 className="card-title">확장 리스크·수익 지표</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(5, 1fr)",
          gap: 16,
        }}
      >
        {items.map((it) => (
          <div key={it.label}>
            <div style={{ color: "var(--muted)", fontSize: 12 }}>{it.label}</div>
            <div
              style={{
                fontSize: 20,
                fontWeight: 700,
                marginTop: 4,
                color:
                  it.tone === "positive"
                    ? "var(--positive)"
                    : it.tone === "negative"
                    ? "var(--negative)"
                    : "var(--text)",
              }}
            >
              {it.value}
            </div>
            <div style={{ fontSize: 11, color: "var(--sub)", marginTop: 2 }}>{it.hint}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function pct(v: number): string {
  const s = v >= 0 ? "+" : "";
  return `${s}${(v * 100).toFixed(2)}%`;
}
