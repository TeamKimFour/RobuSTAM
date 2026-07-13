import Link from "next/link";

export default function TopNav() {
  return (
    <header
      style={{
        borderBottom: "1px solid var(--border)",
        padding: "16px 32px",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
      }}
    >
      <div style={{ display: "flex", alignItems: "center", gap: 32 }}>
        <Link href="/" style={{ fontSize: 18, fontWeight: 700 }}>
          RobuSTAM
        </Link>
        <nav style={{ display: "flex", gap: 24 }}>
          <Link
            href="/dashboard"
            style={{ color: "var(--muted)", fontSize: 14, paddingBottom: 4 }}
          >
            Backtest
          </Link>
          <Link
            href="/dashboard"
            style={{
              color: "var(--text)",
              fontSize: 14,
              borderBottom: "2px solid var(--accent)",
              paddingBottom: 4,
              fontWeight: 500,
            }}
          >
            Inference
          </Link>
        </nav>
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 16 }}>
        <span
          style={{
            fontSize: 12,
            padding: "6px 12px",
            background: "var(--surface-elev)",
            border: "1px solid var(--border)",
            borderRadius: 999,
            color: "var(--muted)",
          }}
        >
          PPO · 5 Assets · Daily Rebalance
        </span>
        <div
          style={{
            width: 32,
            height: 32,
            borderRadius: 999,
            border: "1px solid var(--border)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            color: "var(--muted)",
            fontSize: 14,
          }}
        >
          ◯
        </div>
      </div>
    </header>
  );
}
