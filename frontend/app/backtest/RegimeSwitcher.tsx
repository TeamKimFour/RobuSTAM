"use client";

import { useRouter, useSearchParams } from "next/navigation";
import type { CSSProperties } from "react";
import { REGIMES, type RegimeId } from "./regime";

type Props = { current: RegimeId };

/**
 * 국면 격자 스위처 — 4개 국면(전 구간 · COVID · 고금리 · 최근) 세그먼트 컨트롤.
 *
 * URL 쿼리(`?regime=fold1`)로 상태를 관리해 SSR·공유 URL과 자연스럽게 맞물린다.
 * 서버 컴포넌트(`page.tsx`)가 searchParams에서 값을 읽어 데이터를 필터한 뒤
 * 이미 있는 컴포넌트들에 그대로 넘긴다 — 자식 컴포넌트는 국면 개념을 몰라도 된다.
 */
export default function RegimeSwitcher({ current }: Props) {
  const router = useRouter();
  const searchParams = useSearchParams();

  const activate = (id: RegimeId) => {
    const next = new URLSearchParams(searchParams.toString());
    if (id === "all") {
      next.delete("regime");
    } else {
      next.set("regime", id);
    }
    const qs = next.toString();
    router.replace(qs ? `/backtest?${qs}` : "/backtest", { scroll: false });
  };

  return (
    <div role="tablist" aria-label="시장 국면 필터" style={WRAP}>
      <span style={LABEL}>국면</span>
      <div style={GROUP}>
        {REGIMES.map((r) => {
          const active = r.id === current;
          return (
            <button
              key={r.id}
              type="button"
              role="tab"
              aria-selected={active}
              onClick={() => activate(r.id)}
              title={r.description}
              style={{ ...BTN, ...(active ? BTN_ACTIVE : BTN_IDLE) }}
            >
              {r.shortLabel}
            </button>
          );
        })}
      </div>
    </div>
  );
}

const WRAP: CSSProperties = {
  display: "flex",
  alignItems: "center",
  gap: 12,
};

const LABEL: CSSProperties = {
  fontSize: 12,
  color: "var(--muted)",
  letterSpacing: "0.05em",
  textTransform: "uppercase",
};

const GROUP: CSSProperties = {
  display: "inline-flex",
  padding: 3,
  background: "rgba(255,255,255,0.03)",
  border: "1px solid var(--border)",
  borderRadius: 8,
};

const BTN: CSSProperties = {
  border: "none",
  padding: "6px 12px",
  fontSize: 12,
  fontWeight: 500,
  cursor: "pointer",
  borderRadius: 6,
  transition: "background 0.12s, color 0.12s",
};

const BTN_ACTIVE: CSSProperties = {
  background: "rgba(139,92,246,0.20)",
  color: "#c4b5fd",
};

const BTN_IDLE: CSSProperties = {
  background: "transparent",
  color: "var(--sub)",
};
