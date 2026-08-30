/**
 * 학습 곡선 안내 카드 — SB3 `model.learn()`에 MLflow 콜백이 걸려 있지 않아 스텝별
 * reward/entropy/KL divergence가 기록되지 않는다. 채워 넣을 소스가 없는 자리를 mock으로
 * 위장하지 않고 "미배선" 상태를 그대로 노출한다.
 *
 * 원본 Plotly 3축 차트는 `git log`에 남아 있으며, train.py에 MLflow 콜백이 추가되는
 * 후속 PR에서 되살릴 수 있다. 그때는 필요한 필드(mock.ts::LearningCurvePoint)를
 * 다시 mock.ts에 정의하고 이 파일을 원래 차트로 되돌리면 된다.
 */

export default function LearningCurveChart() {
  return (
    <section className="card">
      <h2 className="card-title">배포 모델 학습 곡선</h2>
      <div
        style={{
          padding: "20px 22px",
          background: "rgba(245,158,11,0.06)",
          border: "1px dashed rgba(245,158,11,0.4)",
          borderRadius: 8,
          color: "#fbbf24",
          fontSize: 13,
          lineHeight: 1.6,
        }}
      >
        <b>MLflow 콜백 미배선</b> — SB3 <code>model.learn()</code>에 스텝별 로그를 남기는 콜백이
        아직 걸려 있지 않아, <b>reward · entropy · KL divergence</b> 시계열이 기록되지 않습니다.
        <br />
        train.py에 콜백이 추가되면 이 자리를 원래 3축 차트로 되돌립니다 (도현 담당).
      </div>
    </section>
  );
}
