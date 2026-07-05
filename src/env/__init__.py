"""자산 배분용 커스텀 Gymnasium 환경.

State/Action 계약: docs/state_spec.md (§0~§4).
차원·자산 순서·거래비용은 config.yaml에서 읽어 조립한다 (하드코딩 금지).
"""

from src.env.portfolio_env import PortfolioEnv

__all__ = ["PortfolioEnv"]
