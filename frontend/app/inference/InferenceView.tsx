"use client";

import { useCallback, useEffect, useState } from "react";
import AllocationCard from "../components/AllocationCard";
import ApiStatusCard from "../components/ApiStatusCard";
import DailyChangeCard from "../components/DailyChangeCard";
import DemoBadge from "../components/DemoBadge";
import FeaturesPanel from "../components/FeaturesPanel";
import HistoryTable from "../components/HistoryTable";
import InferenceMetaCard from "../components/InferenceMetaCard";
import { fetchLatestInference, type ApiResult, type LatestInference } from "../lib/api";

type State =
  | { status: "loading" }
  | { status: "ready"; data: LatestInference; latencyMs: number }
  | { status: "error"; error: Exclude<ApiResult<LatestInference>, { ok: true }>["error"] };

export default function InferenceView() {
  const [state, setState] = useState<State>({ status: "loading" });

  const load = useCallback(async () => {
    setState({ status: "loading" });
    const started = performance.now();
    const result = await fetchLatestInference();
    const latencyMs = Math.round(performance.now() - started);
    if (result.ok) {
      setState({ status: "ready", data: result.data, latencyMs });
    } else {
      setState({ status: "error", error: result.error });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
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
      <PageHeader onRefresh={load} refreshing={state.status === "loading"} />

      {state.status === "loading" ? (
        <SkeletonRow />
      ) : state.status === "error" ? (
        <ApiStatusCard error={state.error} onRetry={load} />
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr",
            gap: 20,
          }}
        >
          <AllocationCard date={state.data.date} weights={state.data.weights} />
          <InferenceMetaCard
            generatedAt={state.data.generated_at}
            modelVersion={state.data.model_version}
            latencyMs={state.latencyMs}
          />
        </div>
      )}

      <SectionTitle>전일 대비 변동성 <DemoBadge /></SectionTitle>
      <DailyChangeCard />

      <SectionTitle>권장 배분 근거 <DemoBadge /></SectionTitle>
      <FeaturesPanel />

      <SectionTitle>최근 이력 <DemoBadge /></SectionTitle>
      <HistoryTable />

      <footer
        style={{
          textAlign: "center",
          color: "var(--sub)",
          fontSize: 12,
          marginTop: 24,
        }}
      >
        © 2026 RobuSTAM Quantitative RL System. All rights reserved.
      </footer>
    </main>
  );
}

function PageHeader({ onRefresh, refreshing }: { onRefresh: () => void; refreshing: boolean }) {
  return (
    <div
      style={{
        display: "flex",
        justifyContent: "space-between",
        alignItems: "flex-end",
        gap: 12,
      }}
    >
      <div>
        <h1
          style={{
            margin: 0,
            fontSize: 24,
            fontWeight: 700,
            letterSpacing: "-0.02em",
          }}
        >
          Inference
        </h1>
        <p
          style={{
            margin: "6px 0 0 0",
            color: "var(--muted)",
            fontSize: 13,
          }}
        >
          시장 상태 → 익일 최적 자산 비중. FastAPI /inference/latest 실시간 조회.
        </p>
      </div>
      <button
        onClick={onRefresh}
        disabled={refreshing}
        style={{
          padding: "8px 14px",
          background: "var(--surface-elev)",
          color: "var(--text)",
          border: "1px solid var(--border)",
          borderRadius: 8,
          fontSize: 13,
          cursor: refreshing ? "not-allowed" : "pointer",
          opacity: refreshing ? 0.6 : 1,
        }}
      >
        {refreshing ? "불러오는 중..." : "새로고침"}
      </button>
    </div>
  );
}

function SectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <h2
      style={{
        margin: "8px 0 -4px 0",
        fontSize: 14,
        fontWeight: 600,
        color: "var(--muted)",
        letterSpacing: "0.02em",
      }}
    >
      {children}
    </h2>
  );
}

function SkeletonRow() {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "1fr 1fr",
        gap: 20,
      }}
    >
      <SkeletonCard height={340} />
      <SkeletonCard height={340} />
    </div>
  );
}

function SkeletonCard({ height }: { height: number }) {
  return (
    <div
      className="card"
      style={{
        height,
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        color: "var(--muted)",
        fontSize: 13,
      }}
    >
      데이터 불러오는 중...
    </div>
  );
}
