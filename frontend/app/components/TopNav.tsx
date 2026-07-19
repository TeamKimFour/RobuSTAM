"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const NAV_ITEMS = [
  { href: "/backtest", label: "Backtest" },
  { href: "/inference", label: "Inference" },
  { href: "/models", label: "Models" },
] as const;

export default function TopNav() {
  const pathname = usePathname();

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
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href || pathname.startsWith(`${item.href}/`);
            return (
              <Link
                key={item.href}
                href={item.href}
                style={{
                  color: active ? "var(--text)" : "var(--muted)",
                  fontSize: 14,
                  borderBottom: active ? "2px solid var(--accent)" : "2px solid transparent",
                  paddingBottom: 4,
                  fontWeight: active ? 500 : 400,
                }}
              >
                {item.label}
              </Link>
            );
          })}
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
