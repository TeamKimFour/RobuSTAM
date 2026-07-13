import type { CSSProperties } from "react";

type Row = {
  date: string;
  spy: number;
  ewy: number;
  tlt: number;
  gld: number;
  shv: number;
  ret: number;
  vol: number;
};

const ROWS: Row[] = [
  { date: "2026-06-28", spy: 40.0, ewy: 15.0, tlt: 20.0, gld: 15.0, shv: 10.0, ret: 1.24, vol: 0.0 },
  { date: "2026-06-27", spy: 36.8, ewy: 13.9, tlt: 22.4, gld: 14.2, shv: 12.7, ret: 0.88, vol: -0.12 },
  { date: "2026-06-26", spy: 35.0, ewy: 12.0, tlt: 25.0, gld: 15.0, shv: 13.0, ret: -0.45, vol: 0.25 },
  { date: "2026-06-25", spy: 38.2, ewy: 11.5, tlt: 20.1, gld: 18.4, shv: 11.8, ret: 2.11, vol: 0.88 },
];

const HEAD_CELL: CSSProperties = {
  textAlign: "left",
  padding: "12px 8px",
  fontWeight: 400,
  color: "var(--sub)",
  borderBottom: "1px solid var(--border)",
  fontSize: 12,
};

const CELL: CSSProperties = {
  padding: "14px 8px",
  fontSize: 13,
};

function signColor(v: number) {
  if (v > 0) return "var(--positive)";
  if (v < 0) return "var(--negative)";
  return "var(--muted)";
}

function fmtSigned(v: number) {
  const sign = v > 0 ? "+" : "";
  return `${sign}${v.toFixed(2)}%`;
}

export default function HistoryTable() {
  return (
    <section className="card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 8,
        }}
      >
        <h2 className="card-title" style={{ margin: 0 }}>
          최근 30일 이력
        </h2>
        <button
          style={{
            background: "transparent",
            border: "none",
            color: "var(--muted)",
            fontSize: 12,
            cursor: "pointer",
          }}
        >
          CSV 다운로드
        </button>
      </div>
      <table style={{ width: "100%", borderCollapse: "collapse" }}>
        <thead>
          <tr>
            <th style={HEAD_CELL}>날짜</th>
            <th style={HEAD_CELL}>SPY %</th>
            <th style={HEAD_CELL}>EWY %</th>
            <th style={HEAD_CELL}>TLT %</th>
            <th style={HEAD_CELL}>GLD %</th>
            <th style={HEAD_CELL}>SHV %</th>
            <th style={{ ...HEAD_CELL, textAlign: "right" }}>수익률</th>
            <th style={{ ...HEAD_CELL, textAlign: "right" }}>변동폭</th>
          </tr>
        </thead>
        <tbody>
          {ROWS.map((r, i) => (
            <tr
              key={r.date}
              style={{
                borderBottom:
                  i < ROWS.length - 1 ? "1px solid var(--border)" : "none",
              }}
            >
              <td style={CELL}>{r.date}</td>
              <td style={CELL}>{r.spy.toFixed(1)}</td>
              <td style={CELL}>{r.ewy.toFixed(1)}</td>
              <td style={CELL}>{r.tlt.toFixed(1)}</td>
              <td style={CELL}>{r.gld.toFixed(1)}</td>
              <td style={CELL}>{r.shv.toFixed(1)}</td>
              <td
                style={{
                  ...CELL,
                  textAlign: "right",
                  color: signColor(r.ret),
                }}
              >
                {fmtSigned(r.ret)}
              </td>
              <td
                style={{
                  ...CELL,
                  textAlign: "right",
                  color: signColor(r.vol),
                }}
              >
                {fmtSigned(r.vol)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
