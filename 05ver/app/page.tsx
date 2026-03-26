"use client";

import Image from "next/image";
import {
  type ChangeEvent,
  type FormEvent,
  type ReactNode,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import {
  AudioWaveform,
  Check,
  Disc3,
  Gauge,
  Headphones,
  Link2,
  Music2,
  Piano,
  Play,
  SlidersHorizontal,
  Sparkles,
  Upload,
  Waves,
  X,
} from "lucide-react";

type InstrumentKey =
  | "보컬"
  | "기타"
  | "베이스"
  | "피아노"
  | "드럼"
  | "기타(other)";
type CompareTrack = "original" | "resynth";
type SourceMode = "sample" | "upload" | "link";
type RequestStatus = "editing" | "preparing" | "ready";

type EffectMetric = {
  label: string;
  value: number;
  detail: string;
};

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
  year: string;
  durationLabel: string;
  recommendedInstrument: InstrumentKey;
  moodTags: string[];
  highlights: string[];
  effectMetrics: EffectMetric[];
};

type ScoreRow = {
  bar: string;
  chord: string;
  note: string;
  cue: string;
};

type RequestDraft = {
  sourceMode: SourceMode;
  archiveId: string;
  title: string;
  sourceLink: string;
  note: string;
  file: File | null;
  options: string[];
};

type ComparePreviewState = {
  open: boolean;
  activeTrack: CompareTrack;
  isPlaying: boolean;
  progress: number;
  activeSection: string;
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
    summary:
      "보컬 레이어와 패드성 사운드가 중심이 되는 곡으로, 공간계와 넓은 리버브의 특성이 두드러집니다.",
    year: "2025",
    durationLabel: "03:28",
    recommendedInstrument: "보컬",
    moodTags: ["Spatial", "Dream", "Layered"],
    highlights: [
      "도입부 보컬 호흡과 패드 레이어가 부드럽게 겹칩니다.",
      "후렴에서 리버브 tail이 길어지며 입체감이 커집니다.",
      "기타(other) 파트가 공기감을 채우는 구조입니다.",
    ],
    effectMetrics: [
      { label: "드라이브", value: 0.18, detail: "드라이브 성향은 거의 없음" },
      { label: "공간계", value: 0.82, detail: "넓은 리버브와 긴 tail이 두드러짐" },
      { label: "모듈레이션", value: 0.46, detail: "패드 계열에 미세한 흔들림 존재" },
      { label: "톤 밝기", value: 0.51, detail: "중역 중심의 부드러운 질감" },
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
    summary:
      "기타 중심의 밴드 사운드가 분명하게 드러나며, 후렴에서 드라이브 성향과 밀도감이 크게 증가합니다.",
    year: "2024",
    durationLabel: "03:42",
    recommendedInstrument: "기타",
    moodTags: ["Drive", "Band", "Alt"],
    highlights: [
      "후렴 직전 기타 어택이 뚜렷하게 살아납니다.",
      "베이스와 킥의 동기화가 리듬감을 밀어줍니다.",
      "코러스 구간에서 드라이브와 존재감이 동시에 커집니다.",
    ],
    effectMetrics: [
      { label: "드라이브", value: 0.71, detail: "중간 이상의 오버드라이브 성향" },
      { label: "공간계", value: 0.42, detail: "짧은 리버브와 적은 딜레이 사용" },
      { label: "모듈레이션", value: 0.24, detail: "거의 미세한 수준의 코러스 성향" },
      { label: "톤 밝기", value: 0.63, detail: "중고역이 선명한 기타 톤" },
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
    summary:
      "멜로디와 공간감, 파형 구조의 균형이 좋은 샘플로, 기타 파트와 보컬의 레이어 관계를 보기 좋습니다.",
    year: "2026",
    durationLabel: "03:42",
    recommendedInstrument: "기타",
    moodTags: ["Cinematic", "Ambient", "Hybrid"],
    highlights: [
      "도입부 멜로디와 리버브 밸런스가 안정적입니다.",
      "브리지에서 공간계와 모듈레이션이 동시에 부각됩니다.",
      "재합성 비교용 샘플로 쓰기 좋은 구조입니다.",
    ],
    effectMetrics: [
      { label: "드라이브", value: 0.62, detail: "약한 오버드라이브 계열" },
      { label: "공간계", value: 0.74, detail: "중간 이상 리버브 + 짧은 딜레이" },
      { label: "모듈레이션", value: 0.31, detail: "아주 미세한 코러스 성향" },
      { label: "톤 밝기", value: 0.57, detail: "중고역이 선명한 편" },
    ],
  },
];

const features = [
  {
    icon: Music2,
    title: "악보와 코드",
    description:
      "멜로디, 반주, 코드 진행을 악기별로 정리해 연주 가능한 형태로 탐색합니다.",
  },
  {
    icon: Headphones,
    title: "악기 분리",
    description:
      "보컬, 기타, 베이스, 드럼, 피아노 등 주요 파트를 분리해 개별적으로 분석합니다.",
  },
  {
    icon: SlidersHorizontal,
    title: "기타 이펙터",
    description:
      "드라이브, 공간계, 모듈레이션 성향과 톤 특성을 구간별로 추정합니다.",
  },
  {
    icon: Gauge,
    title: "정확도와 비교",
    description:
      "재합성 음원과 원음원을 비교하고, 유사도와 신뢰도 지표를 함께 제공합니다.",
  },
];

const instrumentContents: Record<
  InstrumentKey,
  { title: string; scoreRows: ScoreRow[] }
> = {
  보컬: {
    title: "보컬 멜로디와 프레이징",
    scoreRows: [
      { bar: "01", chord: "Am9", note: "긴 호흡의 도입 멜로디", cue: "잔향이 길게 유지됨" },
      {
        bar: "02",
        chord: "Fmaj7",
        note: "가사 시작 구간의 완만한 상승",
        cue: "호흡 전환이 명확함",
      },
      { bar: "03", chord: "C", note: "중역대 중심의 안정된 진행", cue: "레이어 보컬이 얇게 겹침" },
      {
        bar: "04",
        chord: "G",
        note: "후렴 진입 직전 볼륨 상승",
        cue: "배킹 보컬이 넓게 펼쳐짐",
      },
    ],
  },
  기타: {
    title: "기타 악보와 코드 진행",
    scoreRows: [
      { bar: "01", chord: "Am9", note: "서스테인 멜로디 라인", cue: "리버브가 넓게 유지됨" },
      {
        bar: "02",
        chord: "Fmaj7",
        note: "보컬과 기타 아르페지오 겹침",
        cue: "코러스가 미세하게 강조됨",
      },
      { bar: "03", chord: "C", note: "베이스 하행 진행", cue: "드라이브가 아주 약하게 들어감" },
      {
        bar: "04",
        chord: "G",
        note: "후렴 진입 직전 텐션 상승",
        cue: "딜레이 테일이 길어짐",
      },
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
      {
        bar: "02",
        chord: "Fmaj7",
        note: "상성부 멜로디를 얇게 받쳐줌",
        cue: "중역대가 넓어짐",
      },
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

const analysisOptions = [
  "코드",
  "멜로디",
  "악보",
  "강약",
  "연주 기법",
  "이펙터",
];

const requestStages = [
  { title: "입력 확인", detail: "선택한 소스와 옵션을 작업창 기본값으로 정리합니다." },
  { title: "분석 창 배치", detail: "악보, 비교, 악기 패널을 현재 샘플 구조에 맞춰 구성합니다." },
  { title: "인터랙션 연결", detail: "버튼과 상세 창이 현재 워크스페이스 문맥에 연결됩니다." },
  { title: "진입 준비 완료", detail: "백엔드 없이도 이어서 붙일 수 있는 상태로 창을 마무리합니다." },
];

const focusableSelectors =
  'a[href], button:not([disabled]), textarea:not([disabled]), input:not([disabled]):not([type="hidden"]), select:not([disabled]), [tabindex]:not([tabindex="-1"])';

function createRequestDraft(
  archiveId = archiveItems[2].id,
  sourceMode: SourceMode = "sample",
): RequestDraft {
  return {
    sourceMode,
    archiveId,
    title: sourceMode === "sample" ? getArchiveById(archiveId).title : "",
    sourceLink: "",
    note: "",
    file: null,
    options: ["코드", "멜로디", "악보", "이펙터"],
  };
}

function getArchiveById(id: string) {
  return archiveItems.find((item) => item.id === id) ?? archiveItems[0];
}

function getRequestTitle(draft: RequestDraft) {
  if (draft.title.trim()) {
    return draft.title.trim();
  }
  if (draft.sourceMode === "upload" && draft.file) {
    return stripFileExtension(draft.file.name);
  }
  if (draft.sourceMode === "link" && draft.sourceLink.trim()) {
    return draft.sourceLink.trim();
  }
  return getArchiveById(draft.archiveId).title;
}

function stripFileExtension(fileName: string) {
  return fileName.replace(/\.[^/.]+$/, "");
}

function formatClock(progress: number, durationInSeconds: number) {
  const seconds = Math.round((progress / 100) * durationInSeconds);
  const minutes = Math.floor(seconds / 60);
  const remaining = seconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remaining).padStart(2, "0")}`;
}

function summarizeText(value: string, maxLength: number) {
  if (value.length <= maxLength) {
    return value;
  }

  return `${value.slice(0, Math.max(0, maxLength - 3))}...`;
}

export default function MusicAnalysisMainPage() {
  const [selectedArchiveId, setSelectedArchiveId] = useState(archiveItems[2].id);
  const [selectedInstrument, setSelectedInstrument] = useState<InstrumentKey>("기타");
  const [requestDialogOpen, setRequestDialogOpen] = useState(false);
  const [requestDraft, setRequestDraft] = useState<RequestDraft>(() =>
    createRequestDraft(archiveItems[2].id),
  );
  const [requestStatus, setRequestStatus] = useState<RequestStatus>("editing");
  const [requestStageIndex, setRequestStageIndex] = useState(0);
  const [requestError, setRequestError] = useState("");
  const [workspacePrepared, setWorkspacePrepared] = useState(false);
  const [samplePreviewId, setSamplePreviewId] = useState<string | null>(null);
  const [comparePreview, setComparePreview] = useState<ComparePreviewState>({
    open: false,
    activeTrack: "original",
    isPlaying: false,
    progress: 24,
    activeSection: "코러스",
  });

  const selectedArchive = useMemo(
    () => getArchiveById(selectedArchiveId),
    [selectedArchiveId],
  );
  const selectedScore = instrumentContents[selectedInstrument];
  const requestArchive = useMemo(
    () => getArchiveById(requestDraft.archiveId),
    [requestDraft.archiveId],
  );
  const previewArchive = useMemo(
    () => (samplePreviewId ? getArchiveById(samplePreviewId) : null),
    [samplePreviewId],
  );

  useEffect(() => {
    if (requestStatus !== "preparing") {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      if (requestStageIndex >= requestStages.length - 1) {
        const preparedArchive = getArchiveById(requestDraft.archiveId);
        setSelectedArchiveId(preparedArchive.id);
        setSelectedInstrument(preparedArchive.recommendedInstrument);
        setWorkspacePrepared(true);
        setRequestStatus("ready");
        return;
      }

      setRequestStageIndex((previous) => previous + 1);
    }, requestStageIndex === 0 ? 380 : 520);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, [requestDraft.archiveId, requestStageIndex, requestStatus]);

  useEffect(() => {
    if (!comparePreview.open || !comparePreview.isPlaying) {
      return;
    }

    const intervalId = window.setInterval(() => {
      setComparePreview((previous) => ({
        ...previous,
        progress: previous.progress >= 100 ? 0 : previous.progress + 1.2,
      }));
    }, 180);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [comparePreview.isPlaying, comparePreview.open]);

  const scrollToId = (id: string) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const resetRequestDialogState = () => {
    setRequestStatus("editing");
    setRequestStageIndex(0);
    setRequestError("");
  };

  const openRequestDialog = (
    sourceMode: SourceMode = "sample",
    archiveId = selectedArchiveId,
  ) => {
    setSamplePreviewId(null);
    setRequestDraft(createRequestDraft(archiveId, sourceMode));
    resetRequestDialogState();
    setRequestDialogOpen(true);
  };

  const closeRequestDialog = () => {
    setRequestDialogOpen(false);
    resetRequestDialogState();
  };

  const handleRequestSubmit = (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    if (requestDraft.options.length === 0) {
      setRequestError("최소 한 개 이상의 분석 옵션을 선택해 주세요.");
      return;
    }

    if (requestDraft.sourceMode === "upload" && !requestDraft.file) {
      setRequestError("업로드 모드에서는 음원 파일을 먼저 선택해야 합니다.");
      return;
    }

    if (
      requestDraft.sourceMode === "link" &&
      requestDraft.sourceLink.trim().length === 0
    ) {
      setRequestError("링크 모드에서는 분석할 주소를 입력해야 합니다.");
      return;
    }

    setRequestError("");
    setRequestStageIndex(0);
    setRequestStatus("preparing");
  };

  const handleRequestOptionToggle = (option: string) => {
    setRequestDraft((previous) => {
      const options = previous.options.includes(option)
        ? previous.options.filter((item) => item !== option)
        : [...previous.options, option];

      return { ...previous, options };
    });
  };

  const handleRequestFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const nextFile = event.target.files?.[0] ?? null;
    setRequestDraft((previous) => ({
      ...previous,
      file: nextFile,
      title:
        previous.title.trim() || !nextFile
          ? previous.title
          : stripFileExtension(nextFile.name),
    }));
  };

  const handleSampleActivate = (archiveId: string) => {
    const targetArchive = getArchiveById(archiveId);
    setSelectedArchiveId(targetArchive.id);
    setSelectedInstrument(targetArchive.recommendedInstrument);
    setWorkspacePrepared(true);
    setSamplePreviewId(null);
    scrollToId("analysis");
  };

  const handleCompareCardToggle = (track: CompareTrack) => {
    setComparePreview((previous) => {
      if (previous.open && previous.activeTrack === track) {
        return { ...previous, isPlaying: !previous.isPlaying };
      }

      return {
        ...previous,
        open: true,
        activeTrack: track,
        isPlaying: true,
      };
    });
  };

  const closeComparePreview = () => {
    setComparePreview((previous) => ({
      ...previous,
      open: false,
      isPlaying: false,
    }));
  };

  return (
    <>
      <main className="min-h-screen bg-[#0d0f12] text-white">
        <Header
          onNavigate={scrollToId}
          onStart={() => openRequestDialog("sample")}
        />
        <Hero
          item={selectedArchive}
          selectedInstrument={selectedInstrument}
          workspacePrepared={workspacePrepared}
          onSampleClick={() => scrollToId("archive")}
          onUploadClick={() => openRequestDialog("upload")}
        />
        <ArchiveSection
          selectedArchiveId={selectedArchiveId}
          onPreviewArchive={setSamplePreviewId}
        />
        <FeatureSection />
        <AnalysisShowcase
          selectedArchive={selectedArchive}
          selectedInstrument={selectedInstrument}
          onSelectInstrument={setSelectedInstrument}
          selectedScore={selectedScore}
        />
        <CompareSection
          comparePreview={comparePreview}
          similarity={selectedArchive.similarity}
          onToggleCard={handleCompareCardToggle}
        />
        <Footer
          onNavigate={scrollToId}
          onStart={() => openRequestDialog("sample")}
        />
      </main>

      {requestDialogOpen ? (
        <AnalysisRequestDialog
          draft={requestDraft}
          errorMessage={requestError}
          requestArchive={requestArchive}
          requestStageIndex={requestStageIndex}
          requestStatus={requestStatus}
          selectedTitle={getRequestTitle(requestDraft)}
          onArchiveChange={(archiveId) =>
            setRequestDraft((previous) => ({
              ...previous,
              archiveId,
              title:
                previous.sourceMode === "sample" && !previous.title.trim()
                  ? getArchiveById(archiveId).title
                  : previous.title,
            }))
          }
          onClose={closeRequestDialog}
          onFileChange={handleRequestFileChange}
          onNoteChange={(note) =>
            setRequestDraft((previous) => ({ ...previous, note }))
          }
          onOpenWorkspace={() => {
            setRequestDialogOpen(false);
            scrollToId("analysis");
          }}
          onReset={() => {
            setRequestDraft(createRequestDraft(requestDraft.archiveId, requestDraft.sourceMode));
            resetRequestDialogState();
          }}
          onSourceLinkChange={(sourceLink) =>
            setRequestDraft((previous) => ({ ...previous, sourceLink }))
          }
          onSourceModeChange={(sourceMode) =>
            setRequestDraft((previous) => ({
              ...createRequestDraft(previous.archiveId, sourceMode),
              archiveId: previous.archiveId,
              options: previous.options,
              note: previous.note,
            }))
          }
          onSubmit={handleRequestSubmit}
          onTitleChange={(title) =>
            setRequestDraft((previous) => ({ ...previous, title }))
          }
          onToggleOption={handleRequestOptionToggle}
        />
      ) : null}

      {previewArchive ? (
        <SamplePreviewDialog
          item={previewArchive}
          onClose={() => setSamplePreviewId(null)}
          onOpenRequest={() => openRequestDialog("sample", previewArchive.id)}
          onSelectSample={() => handleSampleActivate(previewArchive.id)}
        />
      ) : null}

      {comparePreview.open ? (
        <ComparePreviewDialog
          activeTrack={comparePreview.activeTrack}
          activeSection={comparePreview.activeSection}
          isPlaying={comparePreview.isPlaying}
          progress={comparePreview.progress}
          selectedArchive={selectedArchive}
          onChangeProgress={(progress) =>
            setComparePreview((previous) => ({ ...previous, progress }))
          }
          onClose={closeComparePreview}
          onSelectSection={(activeSection) =>
            setComparePreview((previous) => ({ ...previous, activeSection }))
          }
          onSelectTrack={(activeTrack) =>
            setComparePreview((previous) => ({ ...previous, activeTrack }))
          }
          onTogglePlayback={() =>
            setComparePreview((previous) => ({
              ...previous,
              isPlaying: !previous.isPlaying,
            }))
          }
        />
      ) : null}
    </>
  );
}

function Header({
  onNavigate,
  onStart,
}: {
  onNavigate: (id: string) => void;
  onStart: () => void;
}) {
  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#0d0f12]/80 backdrop-blur-xl">
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-8 py-5">
        <button
          type="button"
          onClick={() => onNavigate("top")}
          className="flex items-center gap-3 text-lg font-semibold tracking-[0.28em] text-white/92"
        >
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
          type="button"
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
    <button
      type="button"
      onClick={onClick}
      className="group relative transition hover:text-white"
    >
      {label}
      <span className="absolute -bottom-2 left-0 h-px w-0 bg-fuchsia-300 transition-all duration-300 group-hover:w-full" />
    </button>
  );
}

function Hero({
  item,
  selectedInstrument,
  workspacePrepared,
  onSampleClick,
  onUploadClick,
}: {
  item: ArchiveItem;
  selectedInstrument: InstrumentKey;
  workspacePrepared: boolean;
  onSampleClick: () => void;
  onUploadClick: () => void;
}) {
  return (
    <section id="top" className="relative overflow-hidden border-b border-white/10">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(217,70,239,0.14),transparent_30%),radial-gradient(circle_at_bottom_right,rgba(255,255,255,0.06),transparent_28%)]" />
      <div className="absolute inset-x-0 bottom-0 h-[340px] bg-[linear-gradient(to_top,rgba(255,255,255,0.02),transparent)]" />

      <div className="relative mx-auto grid min-h-[88vh] max-w-[1600px] items-end gap-12 px-8 pb-20 pt-24 lg:grid-cols-[1.05fr_0.95fr]">
        <div className="max-w-[800px]">
          <p className="mb-6 text-sm uppercase tracking-[0.35em] text-white/42">
            Sound · Score · Structure
          </p>
          <h1 className="text-5xl font-semibold leading-[0.94] tracking-[-0.05em] text-white sm:text-7xl lg:text-[104px]">
            음원을 읽고,
            <br />
            악보와 악기로
            <br />
            다시 보여줍니다
          </h1>
          <p className="mt-8 max-w-[650px] text-lg leading-8 text-white/68 sm:text-xl">
            코드, 멜로디, 악기 분리, 기타 이펙터, 재합성 비교까지. 한 곡의 구조와
            질감을 연주자의 시선으로 해석하는 음악 분석 아카이브 서비스.
          </p>

          <div className="mt-10 flex flex-wrap gap-4">
            <button
              type="button"
              onClick={onSampleClick}
              className="rounded-full bg-white px-7 py-3 text-sm font-medium text-[#0f1113] transition hover:bg-white/92"
            >
              샘플 분석 보기
            </button>
            <button
              type="button"
              onClick={onUploadClick}
              className="rounded-full border border-white/15 px-7 py-3 text-sm text-white/82 transition hover:border-white/30 hover:text-white"
            >
              음원 업로드
            </button>
          </div>
        </div>

        <div className="grid gap-5">
          <div className="rounded-[30px] border border-white/10 bg-white/5 p-7 backdrop-blur-sm">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-xs uppercase tracking-[0.28em] text-white/42">
                  Current analysis
                </p>
                <h3 className="mt-3 text-2xl font-medium text-white">
                  {item.title} · {selectedInstrument} 파트
                </h3>
              </div>
              <div className="rounded-full border border-fuchsia-300/20 bg-fuchsia-400/10 px-4 py-1.5 text-sm text-fuchsia-100">
                {workspacePrepared ? "작업 창 준비됨" : "샘플 워크스페이스"}
              </div>
            </div>

            <div className="mt-8 space-y-4">
              <div className="flex items-end gap-4">
                <span className="text-sm text-white/42">BPM</span>
                <span className="text-3xl font-semibold tracking-[-0.04em] text-white">
                  {item.bpm}
                </span>
              </div>
              <div className="flex items-end gap-4">
                <span className="text-sm text-white/42">KEY</span>
                <span className="text-3xl font-semibold tracking-[-0.04em] text-white">
                  {item.key}
                </span>
              </div>
              <div className="flex items-end gap-4">
                <span className="text-sm text-white/42">유사도</span>
                <span className="text-3xl font-semibold tracking-[-0.04em] text-fuchsia-200">
                  {item.similarity.toFixed(2)}
                </span>
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
                  {Array.from({ length: 48 }).map((_, index) => (
                    <span
                      key={index}
                      className="w-full rounded-full bg-gradient-to-t from-fuchsia-300/20 via-fuchsia-200/45 to-white/70"
                      style={{ height: `${28 + ((index * 17) % 46)}%` }}
                    />
                  ))}
                </div>
              </div>
              <div className="grid grid-cols-5 gap-2 text-[11px] uppercase tracking-[0.24em] text-white/42">
                {timeline.map((segment) => (
                  <div
                    key={segment.label}
                    className="rounded-full border border-white/10 px-3 py-2 text-center"
                  >
                    {segment.label}
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
  onPreviewArchive,
}: {
  selectedArchiveId: string;
  onPreviewArchive: (archiveId: string) => void;
}) {
  return (
    <section id="archive" className="mx-auto max-w-[1600px] px-8 py-24">
      <div className="mb-12 flex items-end justify-between gap-6">
        <div>
          <p className="text-sm uppercase tracking-[0.32em] text-white/42">
            Sample Archive
          </p>
          <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
            미리 분석된 음원 아카이브
          </h2>
        </div>
        <p className="max-w-[540px] text-base leading-7 text-white/60">
          다양한 장르와 악기를 포함한 샘플 음원을 통해 코드, 멜로디, 이펙터, 재합성
          비교 결과를 바로 탐색할 수 있습니다.
        </p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        {archiveItems.map((item) => {
          const isSelected = item.id === selectedArchiveId;

          return (
            <article key={item.id}>
              <button
                type="button"
                aria-label={`${item.title} 샘플 보기`}
                onClick={() => onPreviewArchive(item.id)}
                className={`group block w-full overflow-hidden rounded-[28px] border bg-white/5 text-left transition ${
                  isSelected
                    ? "border-fuchsia-300/40 ring-1 ring-fuchsia-300/20"
                    : "border-white/10"
                }`}
              >
                <div className="relative h-[420px] overflow-hidden">
                  <Image
                    fill
                    alt={item.title}
                    className="object-cover transition duration-500 group-hover:scale-105"
                    sizes="(max-width: 1024px) 100vw, 33vw"
                    src={item.image}
                  />
                  <div className="absolute inset-0 bg-gradient-to-t from-[#0d0f12] via-[#0d0f12]/20 to-transparent" />
                  <div className="absolute left-0 right-0 top-0 flex items-center justify-end p-6">
                    {isSelected ? (
                      <span className="rounded-full border border-fuchsia-300/30 bg-fuchsia-300/12 px-3 py-1 text-xs text-fuchsia-100">
                        선택됨
                      </span>
                    ) : null}
                  </div>
                  <div className="absolute bottom-0 left-0 right-0 p-7">
                    <h3 className="text-2xl font-medium text-white">{item.title}</h3>
                    <p className="mt-2 text-sm text-white/68">{item.artist}</p>
                    <p className="mt-4 text-sm text-white/58">{item.genre}</p>
                    <p className="mt-2 text-sm leading-6 text-fuchsia-100/78">
                      {item.focus}
                    </p>
                  </div>
                </div>
              </button>
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
          <p className="text-sm uppercase tracking-[0.32em] text-white/42">
            Analysis Focus
          </p>
          <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
            음악을 이해하는 네 개의 층위
          </h2>
        </div>

        <div className="grid gap-6 lg:grid-cols-4">
          {features.map((item) => {
            const Icon = item.icon;

            return (
              <article
                key={item.title}
                className="rounded-[26px] border border-white/10 bg-white/5 p-7"
              >
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-fuchsia-300/10 text-fuchsia-200">
                  <Icon className="h-5 w-5" />
                </div>
                <h3 className="mt-6 text-2xl font-medium text-white">{item.title}</h3>
                <p className="mt-4 text-sm leading-7 text-white/62">
                  {item.description}
                </p>
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
  selectedScore: { title: string; scoreRows: ScoreRow[] };
}) {
  return (
    <section id="analysis" className="mx-auto max-w-[1600px] px-8 py-24">
      <div className="grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
        <div className="space-y-8">
          <div>
            <p className="text-sm uppercase tracking-[0.32em] text-white/42">
              Instrument View
            </p>
            <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
              악기와 악보로
              <br />
              다시 읽는 한 곡
            </h2>
            <p className="mt-5 text-base leading-7 text-white/58">
              현재 선택된 샘플: {selectedArchive.title} · {selectedArchive.artist}
            </p>
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
                  type="button"
                  aria-pressed={selectedInstrument === item}
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
              {selectedInstrument === "기타"
                ? "기타 이펙터 프로파일"
                : `${selectedInstrument} 파트 프로파일`}
            </div>
            <div className="mt-6 space-y-5">
              {selectedArchive.effectMetrics.map((item) => (
                <div key={item.label}>
                  <div className="flex items-center justify-between text-sm text-white/72">
                    <span>{item.label}</span>
                    <span className="text-fuchsia-100">{item.value.toFixed(2)}</span>
                  </div>
                  <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8">
                    <div
                      className="h-full rounded-full bg-gradient-to-r from-fuchsia-300 via-fuchsia-200 to-white"
                      style={{ width: `${item.value * 100}%` }}
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
                  <p className="text-xs uppercase tracking-[0.28em] text-white/42">
                    Score View
                  </p>
                  <h3 className="mt-2 text-2xl font-medium text-white">
                    {selectedScore.title}
                  </h3>
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
                      <div
                        key={row.bar}
                        className="grid grid-cols-[48px_72px_1fr] items-start gap-4 border-b border-white/6 pb-4 last:border-b-0"
                      >
                        <span className="text-sm text-white/36">{row.bar}</span>
                        <span className="font-medium text-fuchsia-100">
                          {row.chord}
                        </span>
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
                    <span>00:00 — {selectedArchive.durationLabel}</span>
                  </div>
                  <div className="flex h-12 overflow-hidden rounded-full bg-white/6">
                    {timeline.map((item) => (
                      <div
                        key={item.label}
                        className={`flex items-center justify-center text-[11px] uppercase tracking-[0.24em] text-[#0d0f12] ${item.accent}`}
                        style={{ width: item.width }}
                      >
                        {item.label}
                      </div>
                    ))}
                  </div>
                  <div className="relative mt-8 h-28 overflow-hidden rounded-[22px] bg-[#0d0f12] px-4 py-4">
                    <div className="flex h-full items-center gap-1.5">
                      {Array.from({ length: 60 }).map((_, index) => (
                        <span
                          key={index}
                          className="w-full rounded-full bg-gradient-to-t from-fuchsia-300/15 via-fuchsia-200/35 to-white/60"
                          style={{ height: `${18 + ((index * 13) % 62)}%` }}
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
  comparePreview,
  similarity,
  onToggleCard,
}: {
  comparePreview: ComparePreviewState;
  similarity: number;
  onToggleCard: (track: CompareTrack) => void;
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
            분석된 결과가 실제 음향적으로 얼마나 가까운지를 유사도 지표와 파형 비교로
            함께 확인합니다.
          </p>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1fr_1fr_0.7fr]">
          <CompareCard
            title="원음원"
            subtitle="Original"
            isPlaying={comparePreview.open && comparePreview.activeTrack === "original" && comparePreview.isPlaying}
            onToggle={() => onToggleCard("original")}
          />
          <CompareCard
            title="재합성 음원"
            subtitle="Resynthesized"
            isPlaying={comparePreview.open && comparePreview.activeTrack === "resynth" && comparePreview.isPlaying}
            onToggle={() => onToggleCard("resynth")}
          />
          <div className="rounded-[30px] border border-white/10 bg-white/5 p-7">
            <p className="text-xs uppercase tracking-[0.28em] text-white/42">Evaluation</p>
            <div className="mt-6 space-y-5">
              <MetricLine label="전체 유사도" value={similarity} />
              <MetricLine label="코드 정확도" value={0.74} />
              <MetricLine label="멜로디 일치도" value={0.69} />
              <MetricLine label="분리 품질" value={0.77} />
            </div>
            <div className="mt-8 rounded-[22px] border border-fuchsia-300/15 bg-fuchsia-400/8 px-5 py-4 text-sm leading-6 text-fuchsia-100/88">
              후렴 구간에서의 공간계 표현과 보컬 레이어는 원곡과 매우 유사하게
              재현되었지만, 저역 베이스의 어택은 다소 부드럽게 추정되었습니다.
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
          type="button"
          aria-label={`${title} 미리듣기 ${isPlaying ? "일시정지" : "열기"}`}
          onClick={onToggle}
          className={`flex h-12 w-12 items-center justify-center rounded-full border border-white/12 bg-white/[0.04] text-white/75 transition hover:text-white ${
            isPlaying ? "ring-2 ring-fuchsia-300/40 text-fuchsia-100" : ""
          }`}
        >
          <Play className="h-4 w-4 fill-current" />
        </button>
      </div>

      <div className="relative mt-8 h-24 overflow-hidden rounded-[22px] bg-white/[0.04] px-4 py-4">
        <div className="flex h-full items-end gap-1.5">
          {Array.from({ length: 54 }).map((_, index) => (
            <span
              key={index}
              className="w-full rounded-full bg-gradient-to-t from-fuchsia-300/18 via-fuchsia-200/32 to-white/65"
              style={{ height: `${20 + ((index * 11) % 58)}%` }}
            />
          ))}
        </div>
      </div>

      <div className="mt-6 flex items-center gap-3 text-sm text-white/52">
        <AudioWaveform className="h-4 w-4" />
        {isPlaying
          ? "비교 미리듣기 창에서 현재 트랙이 재생 중입니다."
          : "재생 버튼을 누르면 비교 전용 창이 열리고 A/B 흐름을 확인할 수 있습니다."}
      </div>
    </article>
  );
}

function MetricLine({ label, value }: { label: string; value: number }) {
  return (
    <div>
      <div className="flex items-center justify-between text-sm text-white/72">
        <span>{label}</span>
        <span className="text-fuchsia-100">{value.toFixed(2)}</span>
      </div>
      <div className="mt-2 h-2 overflow-hidden rounded-full bg-white/8">
        <div
          className="h-full rounded-full bg-gradient-to-r from-fuchsia-300 via-fuchsia-200 to-white"
          style={{ width: `${value * 100}%` }}
        />
      </div>
    </div>
  );
}

function Footer({
  onNavigate,
  onStart,
}: {
  onNavigate: (id: string) => void;
  onStart: () => void;
}) {
  return (
    <footer className="border-t border-white/10 bg-[#0d0f12]">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-8 py-10 text-sm text-white/42 md:flex-row md:items-center md:justify-between">
        <div>
          <p className="font-semibold tracking-[0.26em] text-white/86">MUSIC ARCHIVE</p>
          <p className="mt-2">
            악보, 악기, 구조, 이펙터 관점으로 한 곡을 다시 해석하는 분석 서비스
          </p>
        </div>
        <div className="flex gap-6">
          <button
            type="button"
            onClick={() => onNavigate("archive")}
            className="transition hover:text-white"
          >
            샘플 아카이브
          </button>
          <button
            type="button"
            onClick={onStart}
            className="transition hover:text-white"
          >
            분석 시작
          </button>
          <button
            type="button"
            onClick={() => onNavigate("compare")}
            className="transition hover:text-white"
          >
            비교 보기
          </button>
        </div>
      </div>
    </footer>
  );
}

function DialogShell({
  title,
  description,
  maxWidthClass = "max-w-[1040px]",
  onClose,
  children,
}: {
  title: string;
  description: string;
  maxWidthClass?: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const panelRef = useRef<HTMLDivElement>(null);
  const onCloseRef = useRef(onClose);

  useEffect(() => {
    onCloseRef.current = onClose;
  }, [onClose]);

  useEffect(() => {
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    const panel = panelRef.current;
    const focusableElements =
      panel?.querySelectorAll<HTMLElement>(focusableSelectors) ?? [];

    window.requestAnimationFrame(() => {
      (focusableElements[0] ?? panel)?.focus();
    });

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }

      if (event.key !== "Tab" || focusableElements.length === 0) {
        return;
      }

      const first = focusableElements[0];
      const last = focusableElements[focusableElements.length - 1];

      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, []);

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 px-4 py-6 backdrop-blur-md sm:px-6 lg:px-10">
      <div
        aria-hidden="true"
        className="absolute inset-0"
        onClick={onClose}
      />
      <div
        ref={panelRef}
        aria-describedby="dialog-description"
        aria-labelledby="dialog-title"
        aria-modal="true"
        role="dialog"
        tabIndex={-1}
        className={`relative z-10 max-h-[90vh] w-full overflow-hidden rounded-[32px] border border-white/10 bg-[#101318] shadow-[0_30px_120px_rgba(0,0,0,0.45)] ${maxWidthClass}`}
      >
        <div className="flex items-start justify-between border-b border-white/10 px-6 py-5 sm:px-7">
          <div>
            <p id="dialog-title" className="text-2xl font-medium text-white">
              {title}
            </p>
            <p
              id="dialog-description"
              className="mt-2 max-w-[680px] text-sm leading-6 text-white/58"
            >
              {description}
            </p>
          </div>
          <button
            type="button"
            aria-label="창 닫기"
            onClick={onClose}
            className="flex h-11 w-11 items-center justify-center rounded-full border border-white/10 bg-white/[0.03] text-white/70 transition hover:text-white"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        <div className="max-h-[calc(90vh-96px)] overflow-y-auto">{children}</div>
      </div>
    </div>
  );
}

function AnalysisRequestDialog({
  draft,
  errorMessage,
  requestArchive,
  requestStageIndex,
  requestStatus,
  selectedTitle,
  onArchiveChange,
  onClose,
  onFileChange,
  onNoteChange,
  onOpenWorkspace,
  onReset,
  onSourceLinkChange,
  onSourceModeChange,
  onSubmit,
  onTitleChange,
  onToggleOption,
}: {
  draft: RequestDraft;
  errorMessage: string;
  requestArchive: ArchiveItem;
  requestStageIndex: number;
  requestStatus: RequestStatus;
  selectedTitle: string;
  onArchiveChange: (archiveId: string) => void;
  onClose: () => void;
  onFileChange: (event: ChangeEvent<HTMLInputElement>) => void;
  onNoteChange: (note: string) => void;
  onOpenWorkspace: () => void;
  onReset: () => void;
  onSourceLinkChange: (sourceLink: string) => void;
  onSourceModeChange: (sourceMode: SourceMode) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onTitleChange: (title: string) => void;
  onToggleOption: (option: string) => void;
}) {
  return (
    <DialogShell
      title="분석 요청 창"
      description="지금 단계에서는 백엔드 없이도 버튼과 창 흐름이 자연스럽게 이어지도록 구성했습니다. 이후 실제 API를 붙이면 이 창을 그대로 요청 폼으로 확장할 수 있습니다."
      onClose={onClose}
    >
      <div className="grid gap-6 px-6 py-6 sm:px-7 lg:grid-cols-[1.08fr_0.92fr]">
        <form className="space-y-6" onSubmit={onSubmit}>
          <div className="grid gap-3 sm:grid-cols-3">
            {(
              [
                { key: "sample", label: "샘플 기반", icon: Sparkles },
                { key: "upload", label: "파일 업로드", icon: Upload },
                { key: "link", label: "링크 입력", icon: Link2 },
              ] satisfies { key: SourceMode; label: string; icon: typeof Sparkles }[]
            ).map((item) => {
              const Icon = item.icon;
              const isActive = draft.sourceMode === item.key;

              return (
                <button
                  key={item.key}
                  type="button"
                  aria-pressed={isActive}
                  onClick={() => onSourceModeChange(item.key)}
                  className={`flex items-center justify-center gap-2 rounded-[20px] border px-4 py-4 text-sm transition ${
                    isActive
                      ? "border-fuchsia-300/40 bg-fuchsia-400/10 text-fuchsia-100"
                      : "border-white/10 bg-white/[0.03] text-white/62 hover:text-white"
                  }`}
                >
                  <Icon className="h-4 w-4" />
                  {item.label}
                </button>
              );
            })}
          </div>

          {draft.sourceMode === "sample" ? (
            <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
              <div className="flex items-center justify-between">
                <p className="text-sm font-medium text-white">워크스페이스 기준 샘플</p>
                <span className="text-xs uppercase tracking-[0.24em] text-white/40">
                  Layout Anchor
                </span>
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-3">
                {archiveItems.map((item) => {
                  const isActive = item.id === draft.archiveId;

                  return (
                    <button
                      key={item.id}
                      type="button"
                      onClick={() => onArchiveChange(item.id)}
                      className={`rounded-[22px] border px-4 py-4 text-left transition ${
                        isActive
                          ? "border-fuchsia-300/30 bg-fuchsia-400/10"
                          : "border-white/10 bg-[#0d0f12]"
                      }`}
                    >
                      <p className="text-sm font-medium text-white">{item.title}</p>
                      <p className="mt-2 text-xs leading-5 text-white/48">{item.focus}</p>
                    </button>
                  );
                })}
              </div>
            </div>
          ) : null}

          {draft.sourceMode === "upload" ? (
            <label className="block rounded-[28px] border border-dashed border-white/14 bg-white/[0.03] p-6 text-left transition hover:border-fuchsia-300/30">
              <input
                accept="audio/*"
                className="sr-only"
                type="file"
                onChange={onFileChange}
              />
              <div className="flex items-start gap-4">
                <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-fuchsia-300/10 text-fuchsia-100">
                  <Upload className="h-5 w-5" />
                </div>
                <div>
                  <p className="text-base font-medium text-white">
                    음원 파일을 선택해 주세요
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/56">
                    현재는 실제 업로드 대신 파일명과 상태 흐름을 먼저 연결해 둡니다.
                  </p>
                  <p className="mt-3 text-sm text-fuchsia-100/85">
                    {draft.file ? draft.file.name : "선택된 파일이 아직 없습니다."}
                  </p>
                </div>
              </div>
            </label>
          ) : null}

          {draft.sourceMode === "link" ? (
            <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
              <label className="text-sm font-medium text-white" htmlFor="source-link">
                링크 주소
              </label>
              <input
                id="source-link"
                type="url"
                value={draft.sourceLink}
                onChange={(event) => onSourceLinkChange(event.target.value)}
                placeholder="https://example.com/audio"
                className="mt-3 w-full rounded-2xl border border-white/10 bg-[#0d0f12] px-4 py-3 text-sm text-white outline-none transition placeholder:text-white/28 focus:border-fuchsia-300/40"
              />
              <p className="mt-3 text-sm leading-6 text-white/52">
                나중에 링크 다운로드 파이프라인과 연결할 수 있도록 입력창 흐름을 먼저
                고정해 둡니다.
              </p>
            </div>
          ) : null}

          <div>
            <label className="text-sm font-medium text-white" htmlFor="session-title">
              세션 제목
            </label>
            <input
              id="session-title"
              type="text"
              value={draft.title}
              onChange={(event) => onTitleChange(event.target.value)}
              placeholder={selectedTitle}
              className="mt-3 w-full rounded-2xl border border-white/10 bg-[#0d0f12] px-4 py-3 text-sm text-white outline-none transition placeholder:text-white/28 focus:border-fuchsia-300/40"
            />
          </div>

          <div>
            <p className="text-sm font-medium text-white">출력 옵션</p>
            <div className="mt-3 flex flex-wrap gap-2">
              {analysisOptions.map((option) => {
                const isSelected = draft.options.includes(option);

                return (
                  <button
                    key={option}
                    type="button"
                    aria-pressed={isSelected}
                    onClick={() => onToggleOption(option)}
                    className={`rounded-full border px-4 py-2 text-sm transition ${
                      isSelected
                        ? "border-fuchsia-300/30 bg-fuchsia-400/12 text-fuchsia-100"
                        : "border-white/10 bg-white/[0.03] text-white/62 hover:text-white"
                    }`}
                  >
                    {option}
                  </button>
                );
              })}
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-white" htmlFor="session-note">
              실험 메모
            </label>
            <textarea
              id="session-note"
              rows={4}
              value={draft.note}
              onChange={(event) => onNoteChange(event.target.value)}
              placeholder="예: 후렴 기타 톤과 비교창 중심으로 먼저 확인"
              className="mt-3 w-full rounded-[24px] border border-white/10 bg-[#0d0f12] px-4 py-3 text-sm text-white outline-none transition placeholder:text-white/28 focus:border-fuchsia-300/40"
            />
          </div>

          {errorMessage ? (
            <p className="rounded-2xl border border-rose-300/20 bg-rose-400/8 px-4 py-3 text-sm text-rose-100">
              {errorMessage}
            </p>
          ) : null}

          <div className="flex flex-wrap gap-3">
            <button
              type="submit"
              className="rounded-full bg-white px-6 py-3 text-sm font-medium text-[#0f1113] transition hover:bg-white/92"
            >
              {requestStatus === "preparing"
                ? "작업 창 준비 중..."
                : requestStatus === "ready"
                  ? "준비 완료"
                  : "이 설정으로 준비하기"}
            </button>
            <button
              type="button"
              onClick={onReset}
              className="rounded-full border border-white/12 px-6 py-3 text-sm text-white/70 transition hover:text-white"
            >
              입력 초기화
            </button>
          </div>
        </form>

        <div className="space-y-5">
          {draft.sourceMode === "sample" ? (
            <div className="overflow-hidden rounded-[30px] border border-white/10 bg-[#0d0f12]">
              <div className="relative h-[220px]">
                <Image
                  fill
                  alt={requestArchive.title}
                  className="object-cover"
                  sizes="(max-width: 1024px) 100vw, 30vw"
                  src={requestArchive.image}
                />
                <div className="absolute inset-0 bg-gradient-to-t from-[#0d0f12] via-[#0d0f12]/20 to-transparent" />
                <div className="absolute bottom-0 left-0 right-0 p-6">
                  <p className="text-xs uppercase tracking-[0.24em] text-white/46">
                    Target Workspace
                  </p>
                  <h3 className="mt-2 text-2xl font-medium text-white">
                    {selectedTitle}
                  </h3>
                  <p className="mt-2 text-sm text-white/62">
                    {requestArchive.artist} · {requestArchive.genre}
                  </p>
                </div>
              </div>
              <div className="grid gap-0 border-t border-white/10 sm:grid-cols-3">
                <DetailCell label="Anchor" value={requestArchive.title} />
                <DetailCell
                  label="Instrument"
                  value={requestArchive.recommendedInstrument}
                />
                <DetailCell label="Outputs" value={`${draft.options.length}개`} />
              </div>
            </div>
          ) : (
            <div className="overflow-hidden rounded-[30px] border border-white/10 bg-[#0d0f12]">
              <div className="border-b border-white/10 p-6">
                <div className="flex items-start gap-4">
                  <div className="flex h-14 w-14 items-center justify-center rounded-[20px] bg-fuchsia-400/10 text-fuchsia-100">
                    {draft.sourceMode === "upload" ? (
                      <Upload className="h-5 w-5" />
                    ) : (
                      <Link2 className="h-5 w-5" />
                    )}
                  </div>
                  <div>
                    <p className="text-xs uppercase tracking-[0.24em] text-white/46">
                      {draft.sourceMode === "upload" ? "Upload Source" : "Linked Source"}
                    </p>
                    <h3 className="mt-2 text-2xl font-medium text-white">
                      {selectedTitle ||
                        (draft.sourceMode === "upload"
                          ? "업로드할 음원"
                          : "연결할 음원 링크")}
                    </h3>
                    <p className="mt-2 text-sm leading-6 text-white/58">
                      {draft.sourceMode === "upload"
                        ? "샘플 기준 없이 현재 파일 자체를 작업 시작점으로 사용합니다."
                        : "샘플 기준 없이 현재 링크 자체를 작업 시작점으로 사용합니다."}
                    </p>
                  </div>
                </div>
              </div>
              <div className="grid gap-0 border-t border-white/10 sm:grid-cols-3">
                <DetailCell
                  label="Source"
                  value={
                    draft.sourceMode === "upload"
                      ? summarizeText(draft.file?.name ?? "파일 선택 전", 32)
                      : summarizeText(
                          draft.sourceLink.trim() || "링크 입력 전",
                          40,
                        )
                  }
                  title={
                    draft.sourceMode === "upload"
                      ? draft.file?.name ?? "파일 선택 전"
                      : draft.sourceLink.trim() || "링크 입력 전"
                  }
                  truncate
                />
                <DetailCell
                  label="Mode"
                  value={draft.sourceMode === "upload" ? "파일 업로드" : "링크 입력"}
                />
                <DetailCell label="Outputs" value={`${draft.options.length}개`} />
              </div>
            </div>
          )}

          <div className="rounded-[30px] border border-white/10 bg-white/[0.03] p-6">
            <div className="flex items-center justify-between">
              <p className="text-sm font-medium text-white">준비 상태</p>
              <span className="rounded-full border border-white/10 bg-white/[0.03] px-3 py-1 text-xs text-white/56">
                {requestStatus === "editing"
                  ? "설정 중"
                  : requestStatus === "preparing"
                    ? "연결 중"
                    : "완료"}
              </span>
            </div>

            {requestStatus === "editing" ? (
              <div className="mt-5 rounded-[24px] border border-white/10 bg-[#0d0f12] p-5 text-sm leading-6 text-white/58">
                선택한 소스와 옵션을 기준으로 분석 창의 기본 틀을 먼저 맞춥니다. 실제
                데이터 연결 전에도 버튼, 상세창, 비교창이 같은 문맥으로 움직이게 하는
                단계예요.
              </div>
            ) : null}

            {requestStatus === "preparing" ? (
              <div className="mt-5 space-y-4">
                {requestStages.map((stage, index) => {
                  const stageStatus =
                    index < requestStageIndex
                      ? "done"
                      : index === requestStageIndex
                        ? "current"
                        : "pending";

                  return (
                    <div
                      key={stage.title}
                      className={`rounded-[22px] border px-4 py-4 ${
                        stageStatus === "done"
                          ? "border-fuchsia-300/20 bg-fuchsia-400/8"
                          : stageStatus === "current"
                            ? "border-white/14 bg-[#0d0f12]"
                            : "border-white/8 bg-transparent"
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`flex h-9 w-9 items-center justify-center rounded-full ${
                            stageStatus === "done"
                              ? "bg-fuchsia-300/20 text-fuchsia-100"
                              : stageStatus === "current"
                                ? "bg-white/[0.06] text-white"
                                : "bg-white/[0.04] text-white/34"
                          }`}
                        >
                          {stageStatus === "done" ? (
                            <Check className="h-4 w-4" />
                          ) : (
                            <Disc3
                              className={`h-4 w-4 ${
                                stageStatus === "current" ? "animate-spin" : ""
                              }`}
                            />
                          )}
                        </div>
                        <div>
                          <p className="text-sm font-medium text-white">{stage.title}</p>
                          <p className="mt-1 text-xs leading-5 text-white/48">
                            {stage.detail}
                          </p>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : null}

            {requestStatus === "ready" ? (
              <div className="mt-5 space-y-4">
                <div className="rounded-[24px] border border-fuchsia-300/20 bg-fuchsia-400/10 p-5">
                  <p className="text-sm font-medium text-fuchsia-100">
                    버튼과 창 흐름이 현재 샘플 기준으로 준비되었습니다.
                  </p>
                  <p className="mt-2 text-sm leading-6 text-fuchsia-100/76">
                    이제 실제 백엔드 결과만 붙이면 요청 창에서 작업창으로 이어지는 흐름을
                    그대로 사용할 수 있습니다.
                  </p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <button
                    type="button"
                    onClick={onOpenWorkspace}
                    className="rounded-full bg-white px-6 py-3 text-sm font-medium text-[#0f1113] transition hover:bg-white/92"
                  >
                    작업 창으로 이동
                  </button>
                  <button
                    type="button"
                    onClick={onReset}
                    className="rounded-full border border-white/12 px-6 py-3 text-sm text-white/70 transition hover:text-white"
                  >
                    다시 설정하기
                  </button>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </DialogShell>
  );
}

function DetailCell({
  label,
  value,
  title,
  truncate = false,
}: {
  label: string;
  value: string;
  title?: string;
  truncate?: boolean;
}) {
  return (
    <div className="border-t border-white/10 px-5 py-4 sm:border-l sm:border-t-0 first:sm:border-l-0">
      <p className="text-[11px] uppercase tracking-[0.24em] text-white/38">{label}</p>
      <p
        title={title}
        className={`mt-2 text-sm text-white/70 ${truncate ? "truncate" : ""}`}
      >
        {value}
      </p>
    </div>
  );
}

function SamplePreviewDialog({
  item,
  onClose,
  onOpenRequest,
  onSelectSample,
}: {
  item: ArchiveItem;
  onClose: () => void;
  onOpenRequest: () => void;
  onSelectSample: () => void;
}) {
  return (
    <DialogShell
      title={item.title}
      description="샘플 카드에서 바로 끝나지 않도록, 상세 창에서 이 샘플의 맥락과 다음 액션까지 연결했습니다."
      onClose={onClose}
    >
      <div className="grid gap-6 px-6 py-6 sm:px-7 lg:grid-cols-[1.02fr_0.98fr]">
        <div className="space-y-5">
          <div className="relative h-[360px] overflow-hidden rounded-[30px] border border-white/10">
            <Image
              fill
              alt={item.title}
              className="object-cover"
              sizes="(max-width: 1024px) 100vw, 48vw"
              src={item.image}
            />
            <div className="absolute inset-0 bg-gradient-to-t from-[#0d0f12] via-[#0d0f12]/20 to-transparent" />
            <div className="absolute bottom-0 left-0 right-0 p-6">
              <p className="text-xs uppercase tracking-[0.24em] text-white/46">
                Sample Preview
              </p>
              <h3 className="mt-2 text-3xl font-medium text-white">{item.title}</h3>
              <p className="mt-2 text-sm text-white/66">
                {item.artist} · {item.genre}
              </p>
            </div>
          </div>

          <div className="grid gap-3 sm:grid-cols-3">
            <DetailTile label="Year" value={item.year} />
            <DetailTile label="Duration" value={item.durationLabel} />
            <DetailTile label="Focus" value={item.recommendedInstrument} />
          </div>
        </div>

        <div className="space-y-5">
          <div className="rounded-[30px] border border-white/10 bg-white/[0.03] p-6">
            <p className="text-sm font-medium text-white">샘플 요약</p>
            <p className="mt-4 text-sm leading-7 text-white/60">{item.summary}</p>
            <div className="mt-5 flex flex-wrap gap-2">
              {item.moodTags.map((tag) => (
                <span
                  key={tag}
                  className="rounded-full border border-white/10 bg-[#0d0f12] px-3 py-1 text-xs uppercase tracking-[0.24em] text-white/54"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>

          <div className="rounded-[30px] border border-white/10 bg-white/[0.03] p-6">
            <p className="text-sm font-medium text-white">이 샘플에서 먼저 볼 지점</p>
            <div className="mt-4 space-y-3">
              {item.highlights.map((highlight) => (
                <div
                  key={highlight}
                  className="rounded-[22px] border border-white/10 bg-[#0d0f12] px-4 py-4 text-sm leading-6 text-white/68"
                >
                  {highlight}
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[30px] border border-white/10 bg-[#0d0f12] p-6">
            <p className="text-sm font-medium text-white">빠른 액션</p>
            <div className="mt-5 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={onSelectSample}
                className="rounded-full bg-white px-6 py-3 text-sm font-medium text-[#0f1113] transition hover:bg-white/92"
              >
                이 샘플로 작업 창 열기
              </button>
              <button
                type="button"
                onClick={onOpenRequest}
                className="rounded-full border border-white/12 px-6 py-3 text-sm text-white/70 transition hover:text-white"
              >
                분석 요청 창으로 보내기
              </button>
            </div>
          </div>
        </div>
      </div>
    </DialogShell>
  );
}

function DetailTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-[24px] border border-white/10 bg-white/[0.03] px-5 py-5">
      <p className="text-[11px] uppercase tracking-[0.24em] text-white/38">{label}</p>
      <p className="mt-2 text-sm text-white/70">{value}</p>
    </div>
  );
}

function ComparePreviewDialog({
  activeTrack,
  activeSection,
  isPlaying,
  progress,
  selectedArchive,
  onChangeProgress,
  onClose,
  onSelectSection,
  onSelectTrack,
  onTogglePlayback,
}: {
  activeTrack: CompareTrack;
  activeSection: string;
  isPlaying: boolean;
  progress: number;
  selectedArchive: ArchiveItem;
  onChangeProgress: (progress: number) => void;
  onClose: () => void;
  onSelectSection: (section: string) => void;
  onSelectTrack: (track: CompareTrack) => void;
  onTogglePlayback: () => void;
}) {
  const currentTime = formatClock(progress, 222);

  return (
    <DialogShell
      title="비교 미리듣기 창"
      description="지금은 실제 오디오 없이도 버튼과 플레이어 창의 흐름을 먼저 고정해 둔 상태입니다. 나중에 원음원과 재합성 URL만 연결하면 그대로 확장할 수 있습니다."
      maxWidthClass="max-w-[960px]"
      onClose={onClose}
    >
      <div className="space-y-6 px-6 py-6 sm:px-7">
        <div className="grid gap-3 sm:grid-cols-2">
          {(
            [
              { key: "original", title: "원음원", subtitle: "Original" },
              { key: "resynth", title: "재합성 음원", subtitle: "Resynthesized" },
            ] satisfies { key: CompareTrack; title: string; subtitle: string }[]
          ).map((item) => {
            const isActive = activeTrack === item.key;

            return (
              <button
                key={item.key}
                type="button"
                aria-pressed={isActive}
                onClick={() => onSelectTrack(item.key)}
                className={`rounded-[24px] border px-5 py-4 text-left transition ${
                  isActive
                    ? "border-fuchsia-300/30 bg-fuchsia-400/10"
                    : "border-white/10 bg-white/[0.03]"
                }`}
              >
                <p className="text-xs uppercase tracking-[0.24em] text-white/44">
                  {item.subtitle}
                </p>
                <p className="mt-2 text-lg font-medium text-white">{item.title}</p>
              </button>
            );
          })}
        </div>

        <div className="rounded-[30px] border border-white/10 bg-[#0d0f12] p-6">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.24em] text-white/42">
                Active Sample
              </p>
              <h3 className="mt-2 text-2xl font-medium text-white">
                {selectedArchive.title}
              </h3>
            </div>
            <button
              type="button"
              onClick={onTogglePlayback}
              className="flex items-center gap-3 rounded-full border border-fuchsia-300/20 bg-fuchsia-400/10 px-5 py-3 text-sm text-fuchsia-100 transition hover:border-fuchsia-300/40"
            >
              <Play className="h-4 w-4 fill-current" />
              {isPlaying ? "일시정지" : "재생 시작"}
            </button>
          </div>

          <div className="mt-8">
            <div className="flex items-center justify-between text-sm text-white/56">
              <span>{currentTime}</span>
              <span>{selectedArchive.durationLabel}</span>
            </div>
            <input
              type="range"
              min="0"
              max="100"
              step="0.1"
              value={progress}
              onChange={(event) => onChangeProgress(Number(event.target.value))}
              className="mt-4 h-2 w-full cursor-pointer accent-fuchsia-200"
            />
            <div className="mt-6 flex h-20 items-end gap-1 overflow-hidden rounded-[22px] bg-white/[0.04] px-4 py-4">
              {Array.from({ length: 56 }).map((_, index) => (
                <span
                  key={index}
                  className={`w-full rounded-full bg-gradient-to-t ${
                    activeTrack === "original"
                      ? "from-white/16 via-white/32 to-fuchsia-200/80"
                      : "from-fuchsia-300/18 via-fuchsia-200/35 to-white/65"
                  }`}
                  style={{ height: `${20 + ((index * 9) % 60)}%` }}
                />
              ))}
            </div>
          </div>
        </div>

        <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="rounded-[30px] border border-white/10 bg-white/[0.03] p-6">
            <p className="text-sm font-medium text-white">구간 선택</p>
            <div className="mt-4 flex flex-wrap gap-2">
              {timeline.map((item) => {
                const isActive = activeSection === item.label;

                return (
                  <button
                    key={item.label}
                    type="button"
                    aria-pressed={isActive}
                    onClick={() => onSelectSection(item.label)}
                    className={`rounded-full px-4 py-2 text-sm transition ${
                      isActive
                        ? "border border-fuchsia-300/30 bg-fuchsia-400/12 text-fuchsia-100"
                        : "border border-white/10 bg-[#0d0f12] text-white/64 hover:text-white"
                    }`}
                  >
                    {item.label}
                  </button>
                );
              })}
            </div>
          </div>

          <div className="rounded-[30px] border border-white/10 bg-white/[0.03] p-6">
            <p className="text-sm font-medium text-white">현재 창 상태</p>
            <div className="mt-4 space-y-3 text-sm text-white/64">
              <div className="rounded-[20px] border border-white/10 bg-[#0d0f12] px-4 py-4">
                활성 트랙: {activeTrack === "original" ? "원음원" : "재합성 음원"}
              </div>
              <div className="rounded-[20px] border border-white/10 bg-[#0d0f12] px-4 py-4">
                선택 구간: {activeSection}
              </div>
              <div className="rounded-[20px] border border-white/10 bg-[#0d0f12] px-4 py-4">
                상태: {isPlaying ? "재생 중" : "대기 중"}
              </div>
            </div>
          </div>
        </div>
      </div>
    </DialogShell>
  );
}
