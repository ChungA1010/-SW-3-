import Link from "next/link";

const SYSTEM_STEPS = [
  {
    title: "1. 입력",
    description: "음원 파일 또는 유튜브 링크를 받습니다.",
  },

  {
    title: "2. 기타 분리",
    description: "음원에서 기타 소리를 분리하여 추출합니다.",
  },
  {
    title: "3. 모델 추론",
    description: "AI 모델로 이펙터 값을 추론합니다",
  },
  {
    title: "4. 분석 결과 출력",
    description: "세가지 이펙터 계열(Drive, Phase, Distortion)의 강도를 출력합니다.",
  },
  {
    title: "5. 피드백 제공",
    description: "분석 결과에 기반해 사용자의 연주를 평가하고 피드백을 제공합니다.",
  },
];

export default function OverviewPage() {
  return (
    <main className="archive-shell">
      <header className="archive-header">
        <div className="archive-brand">
          <span className="archive-brand-mark" />
          <div>
            <p className="archive-brand-eyebrow">기타 이펙터</p>
            <h1>개요</h1>
          </div>
        </div>
        <nav className="archive-nav">
          <Link href="/">입력</Link>
          <Link href="/overview">개요</Link>
        </nav>
      </header>

      <section className="archive-hero archive-hero-compact">
        <div className="archive-hero-copy">
          <p className="section-eyebrow">개요</p>
          <h2>
            시스템이
            <br />
            어떻게 동작하는지
          </h2>
        </div>

        <div className="hero-preview-card">
          <div className="hero-preview-panel">
            <p className="section-eyebrow">흐름</p>
            <h3>입력 → 분리 → 추론 → 피드백</h3>
            <div className="hero-preview-list">
              <Preview label="백엔드" value="Django" />
              <Preview label="프론트" value="Next.js" />
              <Preview label="분리" value="Demucs" />
              <Preview label="모델" value="Qwen2-Audio" />
            </div>
          </div>
        </div>
      </section>

      <section className="feature-section">
        <div className="section-heading">
          <h3>동작 단계</h3>
        </div>

        <div className="overview-grid">
          {SYSTEM_STEPS.map((step) => (
            <article className="feature-card" key={step.title}>
              <h4>{step.title}</h4>
              <p>{step.description}</p>
            </article>
          ))}
        </div>
      </section>
    </main>
  );
}

function Preview({ label, value }: { label: string; value: string }) {
  return (
    <div className="preview-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
