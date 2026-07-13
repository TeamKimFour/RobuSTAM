import Link from "next/link";

export default function Landing() {
  return (
    <div className="landing">
      <div className="landing-top">
        <div className="landing-top-item">
          <span className="label">Quantitative RL</span>
          <span className="sub">Portfolio MLOps Platform</span>
        </div>

        <div className="landing-top-item center">
          <span aria-hidden>◐</span>
          <span>Daily Rebalance</span>
        </div>

        <div className="landing-top-item inline">
          <Link href="https://github.com/TeamKimFour/RobuSTAM">GitHub</Link>
          <Link href="#">Notion</Link>
        </div>

        <div className="landing-top-item right">
          <span className="label">Based in Seoul</span>
          <span className="sub">09 15 KST</span>
        </div>
      </div>

      <div className="landing-center">
        <h1 className="brand">
          RobuSTAM<sup>®</sup>
        </h1>
        <nav className="landing-nav">
          <Link href="/dashboard">Backtest</Link>
          <Link href="/dashboard">Inference</Link>
          <Link href="#">Docs</Link>
        </nav>
      </div>

      <div className="landing-bottom">
        <div className="landing-bl">
          <div className="barcode" />
          <span className="landing-copyright">
            © 2026, RobuSTAM. All rights reserved
          </span>
        </div>

        <div className="landing-br">
          <p className="landing-desc">
            거래비용·슬리피지 등 현실 마찰을 정직하게 통제한 상태에서 강화학습
            에이전트가 매 거래일 5자산 비중을 조정한다. 매크로 국면별 유효성과
            한계를 투명하게 규명하는 것이 목표다.
          </p>

          <Link href="/dashboard" className="landing-cta">
            <div className="landing-cta-top">
              <span>Launch</span>
              <span className="landing-cta-arrow">↗</span>
            </div>
            <div className="landing-cta-bottom">
              <span className="landing-cta-title">Enter</span>
              <span className="landing-cta-sub">Dashboard</span>
            </div>
          </Link>
        </div>
      </div>
    </div>
  );
}
