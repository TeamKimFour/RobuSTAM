import type { ApiFailure } from "../lib/api";

type Props = {
  error: ApiFailure;
  onRetry?: () => void;
};

export default function ApiStatusCard({ error, onRetry }: Props) {
  const { title, description, tone } = describe(error);

  return (
    <section
      className="card"
      style={{
        borderColor: tone === "warn" ? "var(--accent)" : "var(--negative)",
      }}
    >
      <h2 className="card-title" style={{ color: "var(--text)" }}>
        {title}
      </h2>
      <p
        style={{
          fontSize: 13,
          lineHeight: 1.7,
          color: "var(--muted)",
          margin: "0 0 12px 0",
          whiteSpace: "pre-wrap",
        }}
      >
        {description}
      </p>
      {onRetry ? (
        <button
          onClick={onRetry}
          style={{
            padding: "8px 14px",
            background: "var(--accent)",
            color: "#fff",
            border: "none",
            borderRadius: 8,
            fontSize: 13,
            cursor: "pointer",
            fontWeight: 500,
          }}
        >
          다시 시도
        </button>
      ) : null}
    </section>
  );
}

function describe(error: ApiFailure): {
  title: string;
  description: string;
  tone: "warn" | "error";
} {
  if (error.kind === "network") {
    return {
      tone: "error",
      title: "API 서버에 연결할 수 없습니다",
      description: `FastAPI 추론 서버가 켜져 있는지 확인해주세요.\n\n실행: uvicorn src.api.main:app --reload\n환경변수: NEXT_PUBLIC_API_BASE_URL (.env.local)\n\n원본 오류: ${error.message}`,
    };
  }
  if (error.kind === "notReady") {
    return {
      tone: "warn",
      title: "오늘의 추천 비중이 아직 준비되지 않았습니다",
      description: `precompute 배치가 아직 실행되지 않았거나 결과 파일이 없습니다.\n\n${error.message}`,
    };
  }
  return {
    tone: "error",
    title: `API 오류 (HTTP ${error.status})`,
    description: error.message || "알 수 없는 오류가 발생했습니다.",
  };
}
