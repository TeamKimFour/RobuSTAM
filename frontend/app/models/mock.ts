export type MlflowRun = {
  runId: string;
  fold: number;
  seed: number;
  timesteps: number;
  validSharpe: number;
  validMdd: number;
  testSharpe: number;
  status: "finished" | "failed" | "running";
  deployed?: boolean;
};

// 12 runs (3 fold × 2 seed × 2 timesteps) — 실제 이슈 #34 상황 반영: 최고값이 -0.13
export const MLFLOW_RUNS: MlflowRun[] = [
  { runId: "ppo_fold1_50db7bc4", fold: 1, seed: 42, timesteps: 200000, validSharpe: -0.13, validMdd: -0.22, testSharpe: -0.08, status: "finished", deployed: true },
  { runId: "ppo_fold1_a3f21c88", fold: 1, seed: 42, timesteps: 100000, validSharpe: -0.31, validMdd: -0.28, testSharpe: -0.19, status: "finished" },
  { runId: "ppo_fold1_9c8b7d21", fold: 1, seed: 7,  timesteps: 200000, validSharpe: -0.42, validMdd: -0.33, testSharpe: -0.25, status: "finished" },
  { runId: "ppo_fold1_112ac309", fold: 1, seed: 7,  timesteps: 100000, validSharpe: -0.55, validMdd: -0.31, testSharpe: -0.31, status: "finished" },
  { runId: "ppo_fold2_ff10ac21", fold: 2, seed: 42, timesteps: 200000, validSharpe: -0.28, validMdd: -0.19, testSharpe: -0.12, status: "finished" },
  { runId: "ppo_fold2_8871bb12", fold: 2, seed: 42, timesteps: 100000, validSharpe: -0.44, validMdd: -0.24, testSharpe: -0.22, status: "finished" },
  { runId: "ppo_fold2_c31c9e2a", fold: 2, seed: 7,  timesteps: 200000, validSharpe: -0.38, validMdd: -0.21, testSharpe: -0.18, status: "finished" },
  { runId: "ppo_fold2_88a20de1", fold: 2, seed: 7,  timesteps: 100000, validSharpe: -0.52, validMdd: -0.27, testSharpe: -0.29, status: "finished" },
  { runId: "ppo_fold3_ef224011", fold: 3, seed: 42, timesteps: 200000, validSharpe: -0.19, validMdd: -0.20, testSharpe: -0.11, status: "finished" },
  { runId: "ppo_fold3_10a339cc", fold: 3, seed: 42, timesteps: 100000, validSharpe: -0.34, validMdd: -0.24, testSharpe: -0.20, status: "finished" },
  { runId: "ppo_fold3_bc10a021", fold: 3, seed: 7,  timesteps: 200000, validSharpe: -0.27, validMdd: -0.22, testSharpe: -0.15, status: "finished" },
  { runId: "ppo_fold3_aa211d09", fold: 3, seed: 7,  timesteps: 100000, validSharpe: -0.48, validMdd: -0.28, testSharpe: -0.27, status: "finished" },
];

// 학습 곡선(mock) — 배포 모델 기준
export const LEARNING_CURVE = Array.from({ length: 50 }, (_, i) => {
  const step = (i + 1) * 4000;
  const reward = -0.8 + 0.7 * (1 - Math.exp(-i / 12)) + (Math.sin(i / 3) * 0.05);
  const entropy = 2.5 - i * 0.03 + (Math.random() - 0.5) * 0.05;
  const klDiv = 0.02 + (Math.random() - 0.5) * 0.005;
  return { step, reward, entropy: Math.max(0.1, entropy), klDiv };
});

export type HyperParam = { key: string; value: string; note?: string };

export const DEPLOYED_HYPERPARAMS: HyperParam[] = [
  { key: "algorithm", value: "PPO", note: "안정성 우선 · CLAUDE.md §3" },
  { key: "policy", value: "MlpPolicy" },
  { key: "total_timesteps", value: "200,000" },
  { key: "learning_rate", value: "3e-4", note: "SB3 default" },
  { key: "n_steps", value: "2048" },
  { key: "batch_size", value: "64" },
  { key: "gamma", value: "0.99" },
  { key: "gae_lambda", value: "0.95" },
  { key: "clip_range", value: "0.2" },
  { key: "action_bound", value: "10.0", note: "env 로짓 래핑" },
  { key: "seed", value: "42" },
];
