import type { CSSProperties } from "react";
import TopNav from "../components/TopNav";
import DemoBadge from "../components/DemoBadge";
import LearningCurveChart from "./LearningCurveChart";
import RunGridHeatmap from "./RunGridHeatmap";
import { DEPLOYED_HYPERPARAMS, MLFLOW_RUNS } from "./mock";

export const metadata = {
  title: "Models · RobuSTAM",
};

const HEAD: CSSProperties = {
  textAlign: "left",
  padding: "10px 8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};
const CELL: CSSProperties = { padding: "12px 8px", fontSize: 13 };

export default function ModelsPage() {
  const sorted = [...MLFLOW_RUNS].sort((a, b) => b.validSharpe - a.validSharpe);
  const deployed = MLFLOW_RUNS.find((r) => r.deployed) ?? MLFLOW_RUNS[0];

  return (
    <>
      <TopNav />
      <main
        style={{
          maxWidth: 1280,
          margin: "0 auto",
          padding: "24px 32px 48px",
          display: "flex",
          flexDirection: "column",
          gap: 20,
        }}
      >
        <header>
          <h1 style={{ margin: 0, fontSize: 24, fontWeight: 700, letterSpacing: "-0.02em" }}>
            Models <DemoBadge>MLflow mock</DemoBadge>
          </h1>
          <p style={{ margin: "6px 0 0 0", color: "var(--muted)", fontSize: 13 }}>
            12개 학습 run 결과와 배포 모델 상세. MLflow API 프록시 전이라 값은 mock.
          </p>
        </header>

        <WarningBanner />

        <div style={{ display: "grid", gridTemplateColumns: "1fr 1.4fr", gap: 20 }}>
          <DeployedCard runId={deployed.runId} sharpe={deployed.validSharpe} />
          <HyperparamsCard />
        </div>

        <LearningCurveChart />
        <RunGridHeatmap />

        <section className="card">
          <h2 className="card-title">실험 랭킹 (valid Sharpe 내림차순)</h2>
          <table style={{ width: "100%", borderCollapse: "collapse" }}>
            <thead>
              <tr>
                <th style={HEAD}>#</th>
                <th style={HEAD}>Run ID</th>
                <th style={HEAD}>Fold</th>
                <th style={{ ...HEAD, textAlign: "right" }}>Seed</th>
                <th style={{ ...HEAD, textAlign: "right" }}>Timesteps</th>
                <th style={{ ...HEAD, textAlign: "right" }}>valid Sharpe</th>
                <th style={{ ...HEAD, textAlign: "right" }}>valid MDD</th>
                <th style={{ ...HEAD, textAlign: "right" }}>test Sharpe</th>
                <th style={HEAD}>배포</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => (
                <tr
                  key={r.runId}
                  style={{
                    borderBottom: "1px solid var(--border)",
                    background: r.deployed ? "rgba(139,92,246,0.08)" : undefined,
                  }}
                >
                  <td style={{ ...CELL, color: "var(--muted)" }}>{i + 1}</td>
                  <td style={{ ...CELL, fontFamily: "ui-monospace, monospace", fontSize: 12 }}>
                    {r.runId}
                  </td>
                  <td style={CELL}>Fold {r.fold}</td>
                  <td style={{ ...CELL, textAlign: "right" }}>{r.seed}</td>
                  <td style={{ ...CELL, textAlign: "right" }}>{r.timesteps.toLocaleString()}</td>
                  <td
                    style={{
                      ...CELL,
                      textAlign: "right",
                      color: r.validSharpe >= 0 ? "var(--positive)" : "var(--negative)",
                      fontWeight: 600,
                    }}
                  >
                    {r.validSharpe.toFixed(2)}
                  </td>
                  <td style={{ ...CELL, textAlign: "right", color: "var(--negative)" }}>
                    {(r.validMdd * 100).toFixed(1)}%
                  </td>
                  <td
                    style={{
                      ...CELL,
                      textAlign: "right",
                      color: r.testSharpe >= 0 ? "var(--positive)" : "var(--negative)",
                    }}
                  >
                    {r.testSharpe.toFixed(2)}
                  </td>
                  <td style={CELL}>
                    {r.deployed ? (
                      <span
                        style={{
                          fontSize: 11,
                          padding: "3px 8px",
                          borderRadius: 999,
                          background: "rgba(139,92,246,0.2)",
                          color: "var(--accent)",
                          fontWeight: 600,
                        }}
                      >
                        DEPLOYED
                      </span>
                    ) : (
                      <span style={{ color: "var(--sub)", fontSize: 12 }}>—</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>

        <footer style={{ textAlign: "center", color: "var(--sub)", fontSize: 12, marginTop: 24 }}>
          © 2026 RobuSTAM Quantitative RL System.
        </footer>
      </main>
    </>
  );
}

function WarningBanner() {
  return (
    <div
      style={{
        padding: "12px 16px",
        background: "rgba(245,158,11,0.08)",
        border: "1px solid rgba(245,158,11,0.35)",
        color: "#fbbf24",
        borderRadius: 8,
        fontSize: 13,
        lineHeight: 1.5,
      }}
    >
      <b>배선 검증 전용:</b> 현재 배포된 정책의 valid Sharpe가 <b>음수(-0.13)</b>입니다.
      과매매로 거래비용이 수익을 잠식 중(이슈 #34). precompute → API 배선이 도는지 확인하는
      용도로만 쓰이며, 실제 투자 판단 근거로 삼지 마세요.
    </div>
  );
}

function DeployedCard({ runId, sharpe }: { runId: string; sharpe: number }) {
  return (
    <section className="card">
      <h2 className="card-title">현재 배포 모델</h2>
      <div style={{ display: "grid", gap: 12 }}>
        <Row label="Run ID" value={runId} mono />
        <Row label="model_version" value="ppo-fold1-50db7bc4" mono />
        <Row label="model_path" value="mlruns/models/ppo_fold1_b461...zip" mono small />
        <Row
          label="valid Sharpe"
          value={sharpe.toFixed(2)}
          tone={sharpe >= 0 ? "positive" : "negative"}
        />
        <Row label="scaler fold" value="Fold 1" />
        <Row label="config hash" value="b461a860" mono />
      </div>
    </section>
  );
}

function HyperparamsCard() {
  return (
    <section className="card">
      <h2 className="card-title">배포 모델 하이퍼파라미터</h2>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <tbody>
          {DEPLOYED_HYPERPARAMS.map((h) => (
            <tr key={h.key} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={{ ...CELL, color: "var(--muted)", width: 150 }}>{h.key}</td>
              <td
                style={{
                  ...CELL,
                  fontFamily: "ui-monospace, monospace",
                  fontSize: 12,
                  fontWeight: 600,
                }}
              >
                {h.value}
              </td>
              <td style={{ ...CELL, color: "var(--sub)", fontSize: 12 }}>{h.note ?? ""}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}

function Row({
  label,
  value,
  mono,
  small,
  tone,
}: {
  label: string;
  value: string;
  mono?: boolean;
  small?: boolean;
  tone?: "positive" | "negative";
}) {
  const color =
    tone === "positive" ? "var(--positive)" : tone === "negative" ? "var(--negative)" : "var(--text)";
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
      <span style={{ color: "var(--muted)", fontSize: 13 }}>{label}</span>
      <span
        style={{
          fontSize: small ? 12 : 14,
          fontWeight: 600,
          color,
          fontFamily: mono ? "ui-monospace, SFMono-Regular, Menlo, monospace" : undefined,
          wordBreak: "break-all",
          textAlign: "right",
        }}
      >
        {value}
      </span>
    </div>
  );
}
