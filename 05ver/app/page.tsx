"use client";

import { useMemo, useState } from "react";
import {
  AudioWaveform,
  Disc3,
  Gauge,
  Headphones,
  Music2,
  Piano,
  Play,
  SlidersHorizontal,
  Sparkles,
  Waves,
} from "lucide-react";

type InstrumentKey = "보컬" | "기타" | "베이스" | "피아노" | "드럼" | "기타(other)";

type ArchiveItem = {
  id: string;
  title: string;
  artist: string;
  genre: string;
  focus: string;
  image: string;
  bpm: number;
  key: string;
  similarity: number;
  summary: string;
  effectMetrics: { label: string; value: string; detail: string }[];
};

const archiveItems: ArchiveItem[] = [
  {
    id: "midnight-tape",
    title: "Midnight Tape",
    artist: "Aster Room",
    genre: "드림팝 · 일렉트로닉",
    focus: "코드 진행 · 공간계 이펙터 · 보컬 레이어",
    image:
      "https://images.unsplash.com/photo-1493225457124-a3eb161ffa5f?auto=format&fit=crop&w=1400&q=80",
    bpm: 118,
    key: "F major",
    similarity: 0.76,
    summary: "보컬 레이어와 패드성 사운드가 중심이 되는 곡으로, 공간계와 넓은 리버브의 특성이 두드러집니다.",
    effectMetrics: [
      { label: "드라이브", value: "0.18", detail: "드라이브 성향은 거의 없음" },
      { label: "공간계", value: "0.82", detail: "넓은 리버브와 긴 tail이 두드러짐" },
      { label: "모듈레이션", value: "0.46", detail: "패드 계열에 미세한 흔들림 존재" },
      { label: "톤 밝기", value: "0.51", detail: "중역 중심의 부드러운 질감" },
    ],
  },
  {
    id: "signal-bloom",
    title: "Signal Bloom",
    artist: "Neon Vale",
    genre: "인디록 · 얼터너티브",
    focus: "기타 톤 분석 · 드라이브 계열 · 베이스 라인",
    image:
      "https://images.unsplash.com/photo-1516280440614-37939bbacd81?auto=format&fit=crop&w=1400&q=80",
    bpm: 132,
    key: "E minor",
    similarity: 0.79,
    summary: "기타 중심의 밴드 사운드가 분명하게 드러나며, 후렴에서 드라이브 성향과 밀도감이 크게 증가합니다.",
    effectMetrics: [
      { label: "드라이브", value: "0.71", detail: "중간 이상의 오버드라이브 성향" },
      { label: "공간계", value: "0.42", detail: "짧은 리버브와 적은 딜레이 사용" },
      { label: "모듈레이션", value: "0.24", detail: "거의 미세한 수준의 코러스 성향" },
      { label: "톤 밝기", value: "0.63", detail: "중고역이 선명한 기타 톤" },
    ],
  },
  {
    id: "glass-echo",
    title: "Glass Echo",
    artist: "Lune Archive",
    genre: "시네마틱 · 앰비언트",
    focus: "멜로디 전사 · 파형 비교 · 재합성 유사도",
    image:
      "https://images.unsplash.com/photo-1501386761578-eac5c94b800a?auto=format&fit=crop&w=1400&q=80",
    bpm: 124,
    key: "A minor",
    similarity: 0.81,
    summary: "멜로디와 공간감, 파형 구조의 균형이 좋은 샘플로, 기타 파트와 보컬의 레이어 관계를 보기 좋습니다.",
    effectMetrics: [
      { label: "드라이브", value: "0.62", detail: "약한 오버드라이브 계열" },
      { label: "공간계", value: "0.74", detail: "중간 이상 리버브 + 짧은 딜레이" },
      { label: "모듈레이션", value: "0.31", detail: "아주 미세한 코러스 성향" },
      { label: "톤 밝기", value: "0.57", detail: "중고역이 선명한 편" },
    ],
  },
];

const features = [
  {
    icon: Music2,
    title: "악보와 코드",
    description: "멜로디, 반주, 코드 진행을 악기별로 정리해 연주 가능한 형태로 탐색합니다.",
  },
  {
    icon: Headphones,
    title: "악기 분리",
    description: "보컬, 기타, 베이스, 드럼, 피아노 등 주요 파트를 분리해 개별적으로 분석합니다.",
  },
  {
    icon: SlidersHorizontal,
    title: "기타 이펙터",
    description: "드라이브, 공간계, 모듈레이션 성향과 톤 특성을 구간별로 추정합니다.",
  },
  {
    icon: Gauge,
    title: "정확도와 비교",
    description: "재합성 음원과 원음원을 비교하고, 유사도와 신뢰도 지표를 함께 제공합니다.",
  },
];

const instrumentContents: Record<InstrumentKey, { title: string; scoreRows: { bar: string; chord: string; note: string; cue: string }[] }> = {
  보컬: {
    title: "보컬 멜로디와 프레이징",
    scoreRows: [
      { bar: "01", chord: "Am9", note: "긴 호흡의 도입 멜로디", cue: "잔향이 길게 유지됨" },
      { bar: "02", chord: "Fmaj7", note: "가사 시작 구간의 완만한 상승", cue: "호흡 전환이 명확함" },
      { bar: "03", chord: "C", note: "중역대 중심의 안정된 진행", cue: "레이어 보컬이 얇게 겹침" },
      { bar: "04", chord: "G", note: "후렴 진입 직전 볼륨 상승", cue: "배킹 보컬이 넓게 펼쳐짐" },
    ],
  },
  기타: {
    title: "기타 악보와 코드 진행",
    scoreRows: [
      { bar: "01", chord: "Am9", note: "서스테인 멜로디 라인", cue: "리버브가 넓게 유지됨" },
      { bar: "02", chord: "Fmaj7", note: "보컬과 기타 아르페지오 겹침", cue: "코러스가 미세하게 강조됨" },
      { bar: "03", chord: "C", note: "베이스 하행 진행", cue: "드라이브가 아주 약하게 들어감" },
      { bar: "04", chord: "G", note: "후렴 진입 직전 텐션 상승", cue: "딜레이 테일이 길어짐" },
    ],
  },
  베이스: {
    title: "베이스 라인과 리듬 포지션",
    scoreRows: [
      { bar: "01", chord: "Am9", note: "루트 중심의 저역 앵커", cue: "어택이 부드럽게 시작됨" },
      { bar: "02", chord: "Fmaj7", note: "하행 연결음 사용", cue: "저역이 과하지 않게 정리됨" },
      { bar: "03", chord: "C", note: "박자 사이를 메우는 연결", cue: "킥과 동기화됨" },
      { bar: "04", chord: "G", note: "후렴 진입 전 긴장감 형성", cue: "다이내믹이 소폭 상승" },
    ],
  },
  피아노: {
    title: "피아노 보이싱과 화성 레이어",
    scoreRows: [
      { bar: "01", chord: "Am9", note: "확장화음 중심의 보이싱", cue: "잔향이 뒤로 퍼짐" },
      { bar: "02", chord: "Fmaj7", note: "상성부 멜로디를 얇게 받쳐줌", cue: "중역대가 넓어짐" },
      { bar: "03", chord: "C", note: "리듬보다 질감 중심", cue: "어택이 둥글게 표현됨" },
      { bar: "04", chord: "G", note: "후렴 연결을 위한 보강", cue: "스테레오 폭이 확대됨" },
    ],
  },
  드럼: {
    title: "드럼 패턴과 강약 구조",
    scoreRows: [
      { bar: "01", chord: "-", note: "킥과 스네어의 최소 패턴", cue: "도입은 얇게 시작" },
      { bar: "02", chord: "-", note: "하이햇 밀도 증가", cue: "프리코러스 전 예열 구간" },
      { bar: "03", chord: "-", note: "킥 포인트가 더 선명해짐", cue: "공간계가 얕게 걸림" },
      { bar: "04", chord: "-", note: "후렴 직전 필인 등장", cue: "다이내믹 피크 형성" },
    ],
  },
  "기타(other)": {
    title: "기타 보조 요소와 배경 레이어",
    scoreRows: [
      { bar: "01", chord: "Pad", note: "앰비언트 레이어 진입", cue: "배경 질감 확장" },
      { bar: "02", chord: "Fx", note: "짧은 노이즈 스웰", cue: "공간감 보강" },
      { bar: "03", chord: "Fx", note: "리버스 계열 전환감", cue: "파트 전환 암시" },
      { bar: "04", chord: "Pad", note: "후렴 전 공기감 유지", cue: "메인 파트와 충돌 없이 유지" },
    ],
  },
};

const timeline = [
  { label: "인트로", width: "18%", accent: "bg-fuchsia-400/80" },
  { label: "벌스", width: "24%", accent: "bg-white/65" },
  { label: "프리코러스", width: "14%", accent: "bg-fuchsia-300/70" },
  { label: "코러스", width: "28%", accent: "bg-white/80" },
  { label: "브리지", width: "16%", accent: "bg-fuchsia-500/70" },
];

export default function MusicAnalysisMainPage() {
  const [selectedArchiveId, setSelectedArchiveId] = useState(archiveItems[2].id);
  const [selectedInstrument, setSelectedInstrument] = useState<InstrumentKey>("기타");
  const [playingCard, setPlayingCard] = useState<"original" | "resynth" | null>(null);

  const selectedArchive = useMemo(
    () => archiveItems.find((item) => item.id === selectedArchiveId) ?? archiveItems[0],
    [selectedArchiveId],
  );

  const selectedScore = instrumentContents[selectedInstrument];

  const scrollToId = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const togglePlay = (type: "original" | "resynth") => {
    setPlayingCard((prev) => (prev === type ? null : type));
  };

  return (
    <main className="min-h-screen bg-[#0d0f12] text-white">
      <Header onNavigate={scrollToId} onStart={() => scrollToId("archive")} />
      <Hero item={selectedArchive} onSampleClick={() => scrollToId("archive")} onUploadClick={() => scrollToId("analysis")} />
      <ArchiveSection
        selectedArchiveId={selectedArchiveId}
        onSelectArchive={(id) => {
          setSelectedArchiveId(id);
          scrollToId("analysis");
        }}
      />
      <FeatureSection />
      <AnalysisShowcase
        selectedArchive={selectedArchive}
        selectedInstrument={selectedInstrument}
        onSelectInstrument={setSelectedInstrument}
        selectedScore={selectedScore}
      />
      <CompareSection playingCard={playingCard} onTogglePlay={togglePlay} similarity={selectedArchive.similarity} />
      <Footer onNavigate={scrollToId} />
    </main>
  );
}

function Header({ onNavigate, onStart }: { onNavigate: (id: string) => void; onStart: () => void }) {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#0d0f12]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-8 py-5">
        <button onClick={() => onNavigate("top")} className="flex items-center gap-3 text-lg font-semibold tracking-[0.28em] text-white/92">
          <Disc3 className="h-5 w-5 text-fuchsia-300" />
          <span>MUSIC ARCHIVE</span>
        </button>

        <nav className="hidden items-center gap-8 text-[15px] text-white/72 lg:flex">
          <NavItem label="아카이브" onClick={() => onNavigate("archive")} />
          <NavItem label="분석 기능" onClick={() => onNavigate("features")} />
          <NavItem label="악기별 탐색" onClick={() => onNavigate("analysis")} />
          <NavItem label="악보" onClick={() => onNavigate("analysis")} />
          <NavItem label="이펙터" onClick={() => onNavigate("analysis")} />
          <NavItem label="비교 분석" onClick={() => onNavigate("compare")} />
        </nav>

        <button
          onClick={onStart}
          className="rounded-full border border-fuchsia-300/30 bg-fuchsia-400/10 px-5 py-2.5 text-sm font-medium text-fuchsia-100 transition hover:border-fuchsia-200/60 hover:bg-fuchsia-300/15"
        >
          분석 시작
        </button>
      </div>
    </header>
  );
}

function NavItem({ label, onClick }: { label: string; onClick: () => void }) {
  return (
    <button onClick={onClick} className="group relative transition hover:text-white">
      {label}
      <span className="absolute -bottom-2 left-0 h-px w-0 bg-fuchsia-300 transition-all duration-300 group-hover:w-full" />
    </button>
  );
}

function Hero({
  item,
  onSampleClick,
  onUploadClick,
}: {
  item: ArchiveItem;
  onSampleClick: () => void;
  onUploadClick: () => void;
}) {
  return (
    <section id="top" className="relative overflow-hidden border-b border-white/10">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(217,70,239,0.14),transparent_30%),radial-gradient(circle_at_bottom_right,rgba(255,255,255,0.06),transparent_28%)]" />
      <div className="absolute inset-x-0 bottom-0 h-[340px] bg-[linear-gradient(to_top,rgba(255,255,255,0.02),transparent)]" />

      <div className="relative mx-auto grid min-h-[88vh] max-w-[1600px] items-end gap-12 px-8 pb-20 pt-24 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="max-w-[800px]">
          <p className="mb-6 text-sm uppercase tracking-[0.35em] text-white/42">Sound · Score · Structure</p>
          <h1 className="text-5xl font-semibold leading-[0.94] tracking-[-0.05em] text-white sm:text-7xl lg:text-[104px]">
            음원을 읽고,
            <br />
            악보와 악기로
            <br />
            다시 보여줍니다
          </h1>
          <p className="mt-8 max-w-[650px] text-lg leading-8 text-white/68 sm:text-xl">
            코드, 멜로디, 악기 분리, 기타 이펙터, 재합성 비교까지. 한 곡의 구조와 질감을 연주자의
            시선으로 해석하는 음악 분석 아카이브 서비스.
          </p>

          <div className="mt-10 flex flex-wrap gap-4">
            <button onClick={onSampleClick} className="rounded-full bg-white px-7 py-3 text-sm font-medium text-[#0f1113] transition hover:bg-white/92">
              샘플 분석 보기
            </button>
            <button onClick={onUploadClick} className="rounded-full border border-white/15 px-7 py-3 text-sm text-white/82 transition hover:border-white/30 hover:text-white">
              음원 업로드
            </button>
          </div>
        </div>

        <div className="grid gap-5">
          <div className="rounded-[30px] border border-white/10 bg-white/5 p-7 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-white/42">Current analysis</p>
                <h3 className="mt-3 text-2xl font-medium text-white">
                  {item.title} · {"기타"} 파트
                </h3>
              </div>
              <div className="rounded-full border border-fuchsia-300/20 bg-fuchsia-400/10 px-4 py-1.5 text-sm text-fuchsia-100">
                분석 완료
              </div>
            </div>

            <div className="mt-8 space-y-4">
              <div className="flex items-end gap-4">
                <span className="text-sm text-white/42">BPM</span>
                <span className="text-3xl font-semibold tracking-[-0.04em] text-white">{item.bpm}</span>
              </div>
              <div className="flex items-end gap-4">
                <span className="text-sm text-white/42">KEY</span>
                <span className="text-3xl font-semibold tracking-[-0.04em] text-white">{item.key}</span>
              </div>
              <div className="flex items-end gap-4">
                <span className="text-sm text-white/42">유사도</span>
                <span className="text-3xl font-semibold tracking-[-0.04em] text-fuchsia-200">{item.similarity.toFixed(2)}</span>
              </div>
            </div>
            <p className="mt-6 text-sm leading-7 text-white/58">{item.summary}</p>
          </div>

          <div className="overflow-hidden rounded-[30px] border border-white/10 bg-[#12151a] p-7">
            <div className="flex items-center gap-3 text-sm text-white/48">
              <AudioWaveform className="h-4 w-4" />
              파형과 구조
            </div>
            <div className="mt-6 space-y-4">
              <div className="relative h-20 overflow-hidden rounded-2xl bg-white/[0.04] px-4 py-3">
                <div className="absolute inset-y-0 left-[6%] w-[1px] bg-fuchsia-300/40" />
                <div className="absolute inset-y-0 left-[52%] w-[1px] bg-fuchsia-300/30" />
                <div className="flex h-full items-end gap-1">
                  {Array.from({ length: 48 }).map((_, i) => (
                    <span
                      key={i}
                      className="w-full rounded-full bg-gradient-to-t from-fuchsia-300/20 via-fuchsia-200/45 to-white/70"
                      style={{ height: `${28 + ((i * 17) % 46)}%` }}
                    />
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-5 gap-2 text-[11px] uppercase tracking-[0.24em] text-white/42">
                {timeline.map((item) => (
                  <div key={item.label} className="rounded-full border border-white/10 px-3 py-2 text-center">
                    {item.label}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function ArchiveSection({
  selectedArchiveId,
  onSelectArchive,
}: {
  selectedArchiveId: string;
  onSelectArchive: (id: string) => void;
}) {
  return (
    <section id="archive" className="mx-auto max-w-[1600px] px-8 py-24">
      <div className="mb-12 flex items-end justify-between gap-6">
        <div>
          <p className="text-sm uppercase tracking-[0.32em] text-white/42">Sample Archive</p>
          <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
            미리 분석된 음원 아카이브
          </h2>
        </div>
        <p className="max-w-[540px] text-base leading-7 text-white/60">
          다양한 장르와 악기를 포함한 샘플 음원을 통해 코드, 멜로디, 이펙터, 재합성 비교 결과를
          바로 탐색할 수 있습니다.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {archiveItems.map((item) => {
          const isSelected = item.id === selectedArchiveId;
          return (
            <article
              key={item.title}
              className={`group overflow-hidden rounded-[28px] border bg-white/5 transition ${
                isSelected ? "border-fuchsia-300/40 ring-1 ring-fuchsia-300/20" : "border-white/10"
              }`}
            >
              <div className="relative h-[420px] overflow-hidden">
                <img
                  src={item.image}
                  alt={item.title}
                  className="h-full w-full object-cover transition duration-500 group-hover:scale-105"
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0d0f12] via-[#0d0f12]/20 to-transparent" />
                <div className="absolute left-0 right-0 top-0 flex items-center justify-between p-6">
                  <span className="inline-flex rounded-full border border-white/15 bg-black/20 px-3 py-1 text-xs uppercase tracking-[0.24em] text-white/58 backdrop-blur">
                    SAMPLE
                  </span>
                  {isSelected && (
                    <span className="rounded-full border border-fuchsia-300/30 bg-fuchsia-300/12 px-3 py-1 text-xs text-fuchsia-100">
                      선택됨
                    </span>
                  )}
                </div>
                <div className="absolute bottom-0 left-0 right-0 p-7">
                  <h3 className="text-2xl font-medium text-white">{item.title}</h3>
                  <p className="mt-2 text-sm text-white/68">{item.artist}</p>
                  <p className="mt-4 text-sm text-white/58">{item.genre}</p>
                  <p className="mt-2 text-sm leading-6 text-fuchsia-100/78">{item.focus}</p>
                  <button
                    onClick={() => onSelectArchive(item.id)}
                    className="mt-5 rounded-full border border-white/15 bg-black/25 px-4 py-2 text-sm text-white/88 transition hover:border-white/30 hover:text-white"
                  >
                    이 샘플 보기
                  </button>
                </div>
              </div>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function FeatureSection() {
  return (
    <section id="features" className="border-y border-white/10 bg-[#12151a]">
      <div className="mx-auto max-w-[1600px] px-8 py-24">
        <div className="mb-12">
          <p className="text-sm uppercase tracking-[0.32em] text-white/42">Analysis Focus</p>
          <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
            음악을 이해하는 네 개의 층위
          </h2>
        </div>

        <div className="grid gap-6 lg:grid-cols-4">
          {features.map((item) => {
            const Icon = item.icon;
            return (
              <article key={item.title} className="rounded-[26px] border border-white/10 bg-white/5 p-7">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-fuchsia-300/10 text-fuchsia-200">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="mt-6 text-2xl font-medium text-white">{item.title}</h3>
                <p className="mt-4 text-sm leading-7 text-white/62">{item.description}</p>
              </article>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function AnalysisShowcase({
  selectedArchive,
  selectedInstrument,
  onSelectInstrument,
  selectedScore,
}: {
  selectedArchive: ArchiveItem;
  selectedInstrument: InstrumentKey;
  onSelectInstrument: (instrument: InstrumentKey) => void;
  selectedScore: { title: string; scoreRows: { bar: string; chord: string; note: string; cue: string }[] };
}) {
  return (
    <section id="analysis" className="mx-auto max-w-[1600px] px-8 py-24">
      <div className="grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
        <div className="space-y-8">
          <div>
            <p className="text-sm uppercase tracking-[0.32em] text-white/42">Instrument View</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
              악기와 악보로
              <br />
              다시 읽는 한 곡
            </h2>
            <p className="mt-5 text-base leading-7 text-white/58">현재 선택된 샘플: {selectedArchive.title} · {selectedArchive.artist}</p>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-white/5 p-7">
            <div className="flex items-center gap-3 text-sm text-white/46">
              <Sparkles className="h-4 w-4 text-fuchsia-200" />
              현재 선택된 파트
            </div>
            <div className="mt-6 flex flex-wrap gap-3">
              {(Object.keys(instrumentContents) as InstrumentKey[]).map((item) => (
                <button
                  key={item}
                  onClick={() => onSelectInstrument(item)}
                  className={`rounded-full px-4 py-2 text-sm transition ${
                    selectedInstrument === item
                      ? "border border-fuchsia-300/30 bg-fuchsia-400/12 text-fuchsia-100"
                      : "border border-white/12 bg-transparent text-white/64 hover:text-white"
                  }`}
                >
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-white/5 p-7">
            <div className="flex items-center gap-3 text-sm text-white/46">
              <Waves className="h-4 w-4 text-fuchsia-200" />
              기타 이펙터 프로파일
            </div>
            <div className="mt-6 space-y-5">
              {selectedArchive.effectMetrics.map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between text-sm text-white/72">
                    <span>{item.label}</span>
                    <span className="text-fuchsia-100">{item.value}</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-fuchsia-300 via-fuchsia-200 to-white"
                      style={{ width: `${Number(item.value) * 100}%` }}
                    />
                  </div>
                  <p className="mt-2 text-xs leading-5 text-white/48">{item.detail}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-8">
          <div className="overflow-hidden rounded-[30px] border border-white/10 bg-[#12151a]">
            <div className="border-b border-white/10 px-7 py-5">
              <div className="flex items-center justify-between gap-6">
                <div>
                  <p className="text-xs uppercase tracking-[0.28em] text-white/42">Score View</p>
                  <h3 className="mt-2 text-2xl font-medium text-white">{selectedScore.title}</h3>
                </div>
                <div className="flex items-center gap-2 rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-white/62">
                  <Piano className="h-4 w-4" />
                  MusicXML Ready
                </div>
              </div>
            </div>

            <div className="grid gap-0 lg:grid-cols-[0.9fr_1.1fr]">
              <div className="border-b border-white/10 p-7 lg:border-b-0 lg:border-r">
                <div className="rounded-[22px] border border-white/10 bg-[#0d0f12] p-6">
                  <div className="flex items-center justify-between text-xs uppercase tracking-[0.24em] text-white/42">
                    <span>표기 미리보기</span>
                    <span>{selectedInstrument}</span>
                  </div>
                  <div className="mt-6 space-y-4 text-white/80">
                    {selectedScore.scoreRows.map((row) => (
                      <div key={row.bar} className="grid grid-cols-[48px_72px_1fr] items-start gap-4 border-b border-white/6 pb-4 last:border-b-0">
                        <span className="text-sm text-white/36">{row.bar}</span>
                        <span className="font-medium text-fuchsia-100">{row.chord}</span>
                        <div>
                          <p className="text-sm leading-6 text-white/78">{row.note}</p>
                          <p className="mt-1 text-xs leading-5 text-white/42">{row.cue}</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              <div className="p-7">
                <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-6">
                  <div className="mb-6 flex items-center justify-between text-xs uppercase tracking-[0.24em] text-white/42">
                    <span>구간 타임라인</span>
                    <span>00:00 — 03:42</span>
                  </div>
                  <div className="flex h-12 overflow-hidden rounded-full bg-white/6">
                    {timeline.map((item) => (
                      <div key={item.label} className={`flex items-center justify-center text-[11px] uppercase tracking-[0.24em] text-[#0d0f12] ${item.accent}`} style={{ width: item.width }}>
                        {item.label}
                      </div>
                    ))}
                  </div>
                  <div className="mt-8 relative h-28 overflow-hidden rounded-[22px] bg-[#0d0f12] px-4 py-4">
                    <div className="flex h-full items-center gap-1.5">
                      {Array.from({ length: 60 }).map((_, i) => (
                        <span
                          key={i}
                          className="w-full rounded-full bg-gradient-to-t from-fuchsia-300/15 via-fuchsia-200/35 to-white/60"
                          style={{ height: `${18 + ((i * 13) % 62)}%` }}
                        />
                      ))}
                    </div>
                    <div className="absolute inset-y-4 left-[42%] w-[2px] rounded-full bg-fuchsia-300/60" />
                  </div>
                  <div className="mt-6 grid gap-3 sm:grid-cols-3">
                    <InfoPill label="코드" value="Am9 → Fmaj7 → C → G" />
                    <InfoPill label="강약" value="중간에서 후렴 직전 상승" />
                    <InfoPill label="주석" value={`${selectedInstrument} 중심 해석`} />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function InfoPill({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[20px] border border-white/10 bg-white/[0.03] px-4 py-4">
      <p className="text-[11px] uppercase tracking-[0.24em] text-white/40">{label}</p>
      <p className="mt-2 text-sm leading-6 text-white/72">{value}</p>
    </div>
  );
}

function CompareSection({
  playingCard,
  onTogglePlay,
  similarity,
}: {
  playingCard: "original" | "resynth" | null;
  onTogglePlay: (type: "original" | "resynth") => void;
  similarity: number;
}) {
  return (
    <section id="compare" className="border-t border-white/10 bg-[#12151a] py-24">
      <div className="mx-auto max-w-[1600px] px-8">
        <div className="mb-12 flex items-end justify-between gap-6">
          <div>
            <p className="text-sm uppercase tracking-[0.32em] text-white/42">Compare</p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
              원음원과 재합성 음원의 비교
            </h2>
          </div>
          <p className="max-w-[520px] text-base leading-7 text-white/60">
            분석된 결과가 실제 음향적으로 얼마나 가까운지를 유사도 지표와 파형 비교로 함께 확인합니다.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1fr_1fr_0.7fr]">
          <CompareCard title="원음원" subtitle="Original" isPlaying={playingCard === "original"} onToggle={() => onTogglePlay("original")} />
          <CompareCard title="재합성 음원" subtitle="Resynthesized" isPlaying={playingCard === "resynth"} onToggle={() => onTogglePlay("resynth")} />
          <div className="rounded-[30px] border border-white/10 bg-white/5 p-7">
            <p className="text-xs uppercase tracking-[0.28em] text-white/42">Evaluation</p>
            <div className="mt-6 space-y-5">
              <MetricLine label="전체 유사도" value={similarity.toFixed(2)} />
              <MetricLine label="코드 정확도" value="0.74" />
              <MetricLine label="멜로디 일치도" value="0.69" />
              <MetricLine label="분리 품질" value="0.77" />
            </div>
            <div className="mt-8 rounded-[22px] border border-fuchsia-300/15 bg-fuchsia-400/8 px-5 py-4 text-sm leading-6 text-fuchsia-100/88">
              후렴 구간에서의 공간계 표현과 보컬 레이어는 원곡과 매우 유사하게 재현되었지만, 저역 베이스의
              어택은 다소 부드럽게 추정되었습니다.
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function CompareCard({
  title,
  subtitle,
  isPlaying,
  onToggle,
}: {
  title: string;
  subtitle: string;
  isPlaying: boolean;
  onToggle: () => void;
}) {
  return (
    <article className="rounded-[30px] border border-white/10 bg-[#0d0f12] p-7">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.28em] text-white/42">{subtitle}</p>
          <h3 className="mt-2 text-2xl font-medium text-white">{title}</h3>
        </div>
        <button
          onClick={onToggle}
          className={`flex h-12 w-12 items-center justify-center rounded-full border border-white/12 bg-white/[0.04] text-white/75 transition hover:text-white ${
            isPlaying ? "ring-2 ring-fuchsia-300/40 text-fuchsia-100" : ""
          }`}
        >
          <Play className="h-4 w-4 fill-current" />
        </button>
      </div>

      <div className="mt-8 relative h-24 overflow-hidden rounded-[22px] bg-white/[0.04] px-4 py-4">
        <div className="flex h-full items-end gap-1.5">
          {Array.from({ length: 54 }).map((_, i) => (
            <span
              key={i}
              className="w-full rounded-full bg-gradient-to-t from-fuchsia-300/18 via-fuchsia-200/32 to-white/65"
              style={{ height: `${20 + ((i * 11) % 58)}%` }}
            />
          ))}
        </div>
      </div>

      <div className="mt-6 flex items-center gap-3 text-sm text-white/52">
        <AudioWaveform className="h-4 w-4" />
        {isPlaying ? "재생 중인 상태를 표시하는 인터랙션입니다." : "파형 비교 데이터와 오디오 플레이어가 이 영역에 연결됩니다."}
      </div>
    </article>
  );
}

function MetricLine({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm text-white/72">
        <span>{label}</span>
        <span className="text-fuchsia-100">{value}</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8">
        <div
          className="h-full rounded-full bg-gradient-to-r from-fuchsia-300 via-fuchsia-200 to-white"
          style={{ width: `${Number(value) * 100}%` }}
        />
      </div>
    </div>
  );
}

function Footer({ onNavigate }: { onNavigate: (id: string) => void }) {
  return (
    <footer className="border-t border-white/10 bg-[#0d0f12]">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-8 py-10 text-sm text-white/42 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="font-semibold tracking-[0.26em] text-white/86">MUSIC ARCHIVE</p>
          <p className="mt-2">악보, 악기, 구조, 이펙터 관점으로 한 곡을 다시 해석하는 분석 서비스</p>
        </div>
        <div className="flex gap-6">
          <button onClick={() => onNavigate("archive")} className="transition hover:text-white">샘플 아카이브</button>
          <button onClick={() => onNavigate("analysis")} className="transition hover:text-white">분석 시작</button>
          <button onClick={() => onNavigate("compare")} className="transition hover:text-white">비교 보기</button>
        </div>
      </div>
    </footer>
  );
}
