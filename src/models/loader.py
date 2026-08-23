"""학습된 policy.zip을 알고리즘에 맞는 SB3 클래스로 읽는다.

`PPO.load(path)`를 하드코딩해두면 DQN policy(이슈 #34 개선안 B, 행동 연속성)를 읽을 때
조용히 깨진다 — SB3는 저장 포맷이 같아 보여도 클래스가 다르면 로드에 실패한다. 소비처
(백테스트 러너·실험 러너·추론 precompute)가 전부 이 함수를 통해 읽도록 모아둔다.

알고리즘은 config `model.algorithm`이 단일 출처다(docs/env_spec.md §5 · model_training.md §7).
"""

from __future__ import annotations

SUPPORTED = ("PPO", "DQN")


def load_policy(model_path: str, algorithm: str = "PPO"):
    """`model_path`의 policy.zip을 `algorithm`에 맞는 SB3 클래스로 로드한다."""
    name = str(algorithm).upper()
    if name == "PPO":
        from stable_baselines3 import PPO

        return PPO.load(model_path)
    if name == "DQN":
        from stable_baselines3 import DQN

        return DQN.load(model_path)
    raise NotImplementedError(f"지원하지 않는 알고리즘입니다: {algorithm!r} (지원: {SUPPORTED})")


def get_algorithm(cfg: dict) -> str:
    """config에서 알고리즘 이름을 읽는다(기본 PPO). 소비처가 대문자 규약을 신경쓰지 않도록 정규화."""
    return str(cfg.get("model", {}).get("algorithm", "PPO")).upper()


def is_discrete(cfg: dict) -> bool:
    """이산 행동(DiscretePortfolioEnv) 경로를 써야 하는 설정인가.

    DQN은 유한 이산 행동만 다루므로 env를 어댑터로 감싸야 하고, 백테스트도 같은 Δ 이전
    규칙으로 비중을 굴려야 한다. 이 판정을 한 곳에 두어 학습·평가·백테스트가 갈리지 않게 한다.
    """
    return get_algorithm(cfg) == "DQN"
