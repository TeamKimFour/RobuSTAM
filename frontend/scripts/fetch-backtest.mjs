#!/usr/bin/env node
/**
 * Vercel prebuild — 실데이터 백테스트 JSON을 원격 URL(S3 등)에서 내려받아 public/에 배치한다.
 *
 * 흐름: 백엔드가 `python -m src.backtest.export --upload-s3`로 S3에 올린 파일을,
 *       Vercel 빌드 시 이 스크립트가 fetch해서 `public/backtest.json`으로 저장한다.
 *       그러면 loader.ts가 그 파일을 읽어 실데이터 대시보드를 그린다.
 *
 * 환경변수:
 *   BACKTEST_JSON_URL  — 다운로드 원본. 예:
 *                        https://{bucket}.s3.{region}.amazonaws.com/{prefix}/frontend/backtest.json
 *                        (미설정 시 조용히 skip → loader.ts가 mock으로 fallback)
 *
 * 실패 정책: **어떤 에러도 빌드를 실패시키지 않는다.** URL 없음·네트워크·HTTP 4xx/5xx·JSON 파싱
 * 오류 어느 것이든 warning만 로그하고 exit 0 — mock으로 정상 배포되어야 데모가 안 끊긴다.
 *
 * 의존성: Node 18+ 표준(fetch·fs·path)만 사용. Vercel Node 20 런타임에서 별도 npm 설치 불필요.
 */

import { promises as fs } from "node:fs";
import path from "node:path";
import process from "node:process";

const URL_ENV = "BACKTEST_JSON_URL";
const OUT_RELATIVE = "public/backtest.json";
const FETCH_TIMEOUT_MS = 15_000;

function log(message) {
  process.stdout.write(`[fetch-backtest] ${message}\n`);
}

function warn(message) {
  process.stderr.write(`[fetch-backtest] ${message}\n`);
}

async function fetchWithTimeout(url) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);
  try {
    return await fetch(url, { signal: controller.signal, cache: "no-store" });
  } finally {
    clearTimeout(timer);
  }
}

function looksValid(bundle) {
  return (
    bundle &&
    typeof bundle === "object" &&
    Array.isArray(bundle.strategies) &&
    bundle.strategies.length > 0
  );
}

async function main() {
  const url = process.env[URL_ENV];
  if (!url) {
    log(`${URL_ENV} 미설정 → skip (loader.ts가 mock으로 fallback)`);
    return;
  }

  log(`다운로드 시작 → ${url}`);
  let response;
  try {
    response = await fetchWithTimeout(url);
  } catch (err) {
    warn(`fetch 실패 (mock으로 fallback): ${err?.message ?? err}`);
    return;
  }

  if (!response.ok) {
    warn(`HTTP ${response.status} ${response.statusText} (mock으로 fallback)`);
    return;
  }

  const text = await response.text();
  let parsed;
  try {
    parsed = JSON.parse(text);
  } catch (err) {
    warn(`JSON 파싱 실패 (mock으로 fallback): ${err?.message ?? err}`);
    return;
  }
  if (!looksValid(parsed)) {
    warn("계약 위반 — strategies 배열 누락/빔 (mock으로 fallback)");
    return;
  }

  const outPath = path.join(process.cwd(), OUT_RELATIVE);
  await fs.mkdir(path.dirname(outPath), { recursive: true });
  const tmpPath = `${outPath}.tmp`;
  await fs.writeFile(tmpPath, text, "utf-8");
  await fs.rename(tmpPath, outPath);
  log(`저장 완료 → ${OUT_RELATIVE} (${text.length.toLocaleString()} bytes)`);
}

main().catch((err) => {
  warn(`예상치 못한 에러 (mock으로 fallback): ${err?.message ?? err}`);
  process.exit(0);
});
