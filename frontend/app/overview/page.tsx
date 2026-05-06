import Link from "next/link";

const SYSTEM_STEPS = [
  {
    title: "1. 입력",
    description: "파일 업로드 또는 유튜브 링크를 받습니다.",
  },
  {
    title: "2. 오디오 확보",
    description: "원본 오디오를 저장하고 분석용으로 준비합니다.",
  },
  {
    title: "3. 기타 분리",
    description: "Demucs로 stem을 분리하고 기타에 가까운 stem을 선택합니다.",
  },
  {
    title: "4. 모델 추론",
    description: "GRU 모델로 이펙터 점수와 구간별 결과를 계산합니다.",
  },
  {
    title: "5. 결과 표시",
    description: "대표 이펙터, 점수, 시계열 결과를 화면에 보여줍니다.",
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
            <h3>입력 → 분리 → 추론 → 결과</h3>
            <div className="hero-preview-list">
              <Preview label="백엔드" value="FastAPI" />
              <Preview label="분리" value="Demucs" />
              <Preview label="모델" value="GRU" />
              <Preview label="프론트" value="Next.js" />
            </div>
          </div>
        </div>
      </section>

      <section className="feature-section">
        <div className="section-heading">
          <p className="section-eyebrow">처리 순서</p>
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
