import type { CSSProperties } from "react";
import type { DayReturn } from "./mock";

const CELL: CSSProperties = { padding: "10px 8px", fontSize: 13 };
const HEAD: CSSProperties = {
  textAlign: "left",
  padding: "8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};

type Props = { best: DayReturn[]; worst: DayReturn[] };

export default function BestWorstDaysCard({ best, worst }: Props) {
  return (
    <section className="card">
      <h2 className="card-title">Best · Worst 10 거래일</h2>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gap: 24,
        }}
      >
        <ReturnTable title="Best" rows={best} positive />
        <ReturnTable title="Worst" rows={worst} positive={false} />
      </div>
    </section>
  );
}

function ReturnTable({ title, rows, positive }: { title: string; rows: DayReturn[]; positive: boolean }) {
  return (
    <div>
      <div
        style={{
          fontSize: 12,
          fontWeight: 600,
          color: positive ? "var(--positive)" : "var(--negative)",
          marginBottom: 6,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
        }}
      >
        {title}
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={HEAD}>날짜</th>
            <th style={{ ...HEAD, textAlign: "right" }}>일간 수익률</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.date} style={{ borderBottom: "1px solid var(--border)" }}>
              <td style={CELL}>{r.date}</td>
              <td
                style={{
                  ...CELL,
                  textAlign: "right",
                  color: r.ret >= 0 ? "var(--positive)" : "var(--negative)",
                  fontWeight: 600,
                }}
              >
                {(r.ret * 100).toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
