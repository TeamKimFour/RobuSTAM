/**
 * Models 대시보드 mock 데이터.
 *
 * **이 파일의 각 필드는 `src/models/train.py`가 실제로 MLflow에 로깅하는 것과 1:1 대응한다.**
 * train.py가 안 남기는 값은 UI에서도 표시하지 않는다 — API 프록시(민지) 붙었을 때
 * "채워 넣을 소스가 없는" 카드를 mock으로 위장하지 않기 위함.
 *
 * 실제 로깅되는 항목(train.py::evaluate + start_run.log_params):
 *   Metrics (evaluate, split="valid"로만 호출됨):
 *     - valid_sharpe            : _annualized_sharpe(rewards) — 스텝별 net 로그보상의 연율화 Sharpe proxy
 *     - valid_total_log_return  : sum(rewards)
 *     - valid_total_txn_cost    : sum(info["cost"]) — 실비용 c_real × turnover
 *     - valid_avg_turnover      : mean(info["turnover"])
 *   Params:
 *     - algorithm, policy, fold_id, feature_store_run_id, feature_set, state_dim,
 *       total_timesteps, window, transaction_cost, train_cost_multiplier, vol_penalty_coef, seed
 *     - learning_rate, target_kl, ent_coef (config에 있을 때만)
 *
 * 로깅 안 되는 것(팀 회의 안건 · 도현 담당):
 *   - valid_mdd            : evaluate에 MDD 계산 로직 자체가 없음
 *   - test_sharpe          : evaluate가 split="valid"로만 호출됨 (test 평가 코드 없음)
 *   - 학습 곡선(reward/entropy/KL) : SB3 model.learn()에 MLflow 콜백 안 걸림
 *   → 위 3종은 UI에서 카드로 별도 노출(LearningCurveNotice·표에서 제거)해 "미배선"임을 명시.
 */

export type MlflowRun = {
  runId: string;
  fold: number;
  seed: number;
  timesteps: number;
  featureSet: string;                // 콤보 이름 (full·M0·M1·M2·M3)
  validSharpe: number;               // valid_sharpe
  validTotalLogReturn: number;       // valid_total_log_return
  validTotalTxnCost: number;         // valid_total_txn_cost
  validAvgTurnover: number;          // valid_avg_turnover
  status: "finished" | "failed" | "running";
  deployed?: boolean;
};

// 12 runs (3 fold × 2 seed × 2 timesteps) — 이슈 #34 상황 반영: 최고값이 -0.13.
// 각 필드는 train.py::evaluate() 반환값과 동일 규약(값 자체는 mock).
export const MLFLOW_RUNS: MlflowRun[] = [
  { runId: "ppo_fold1_50db7bc4", fold: 1, seed: 42, timesteps: 200000, featureSet: "full", validSharpe: -0.13, validTotalLogReturn: -0.021, validTotalTxnCost: 3820, validAvgTurnover: 0.44, status: "finished", deployed: true },
  { runId: "ppo_fold1_a3f21c88", fold: 1, seed: 42, timesteps: 100000, featureSet: "full", validSharpe: -0.31, validTotalLogReturn: -0.048, validTotalTxnCost: 4290, validAvgTurnover: 0.51, status: "finished" },
  { runId: "ppo_fold1_9c8b7d21", fold: 1, seed: 7,  timesteps: 200000, featureSet: "full", validSharpe: -0.42, validTotalLogReturn: -0.062, validTotalTxnCost: 5140, validAvgTurnover: 0.60, status: "finished" },
  { runId: "ppo_fold1_112ac309", fold: 1, seed: 7,  timesteps: 100000, featureSet: "full", validSharpe: -0.55, validTotalLogReturn: -0.089, validTotalTxnCost: 5720, validAvgTurnover: 0.68, status: "finished" },
  { runId: "ppo_fold2_ff10ac21", fold: 2, seed: 42, timesteps: 200000, featureSet: "full", validSharpe: -0.28, validTotalLogReturn: -0.041, validTotalTxnCost: 3510, validAvgTurnover: 0.39, status: "finished" },
  { runId: "ppo_fold2_8871bb12", fold: 2, seed: 42, timesteps: 100000, featureSet: "full", validSharpe: -0.44, validTotalLogReturn: -0.068, validTotalTxnCost: 4160, validAvgTurnover: 0.49, status: "finished" },
  { runId: "ppo_fold2_c31c9e2a", fold: 2, seed: 7,  timesteps: 200000, featureSet: "full", validSharpe: -0.38, validTotalLogReturn: -0.055, validTotalTxnCost: 4520, validAvgTurnover: 0.52, status: "finished" },
  { runId: "ppo_fold2_88a20de1", fold: 2, seed: 7,  timesteps: 100000, featureSet: "full", validSharpe: -0.52, validTotalLogReturn: -0.082, validTotalTxnCost: 5350, validAvgTurnover: 0.63, status: "finished" },
  { runId: "ppo_fold3_ef224011", fold: 3, seed: 42, timesteps: 200000, featureSet: "full", validSharpe: -0.19, validTotalLogReturn: -0.029, validTotalTxnCost: 3140, validAvgTurnover: 0.36, status: "finished" },
  { runId: "ppo_fold3_10a339cc", fold: 3, seed: 42, timesteps: 100000, featureSet: "full", validSharpe: -0.34, validTotalLogReturn: -0.052, validTotalTxnCost: 3980, validAvgTurnover: 0.47, status: "finished" },
  { runId: "ppo_fold3_bc10a021", fold: 3, seed: 7,  timesteps: 200000, featureSet: "full", validSharpe: -0.27, validTotalLogReturn: -0.038, validTotalTxnCost: 3620, validAvgTurnover: 0.41, status: "finished" },
  { runId: "ppo_fold3_aa211d09", fold: 3, seed: 7,  timesteps: 100000, featureSet: "full", validSharpe: -0.48, validTotalLogReturn: -0.075, validTotalTxnCost: 4890, validAvgTurnover: 0.58, status: "finished" },
];

/**
 * 배포 모델 하이퍼파라미터·기본 params — train.py::start_run.log_params가 남기는 값들만.
 * key가 train.py에 정확히 대응되도록 (스네이크케이스) 두어, MLflow 프록시 붙었을 때
 * 그대로 매핑되게 한다.
 */
export type HyperParam = { key: string; value: string; note?: string };

export const DEPLOYED_HYPERPARAMS: HyperParam[] = [
  { key: "algorithm", value: "PPO", note: "안정성 우선 · CLAUDE.md §3" },
  { key: "policy", value: "MlpPolicy" },
  { key: "feature_set", value: "full", note: "docs/state_spec.md §2 v1.2 콤보 카탈로그" },
  { key: "state_dim", value: "187", note: "(5×30)+37" },
  { key: "window", value: "30" },
  { key: "transaction_cost", value: "0.001", note: "편도 0.1%" },
  { key: "total_timesteps", value: "200,000" },
  { key: "train_cost_multiplier", value: "1.0", note: "λ (개선안 A)" },
  { key: "vol_penalty_coef", value: "0.0", note: "κ (개선안 D)" },
  { key: "learning_rate", value: "3e-4", note: "SB3 default" },
  { key: "seed", value: "42" },
];
