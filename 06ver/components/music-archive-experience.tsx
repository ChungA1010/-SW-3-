"use client";

import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import {
  type ChangeEvent,
  type CSSProperties,
  type FormEvent,
  type ReactNode,
  type RefObject,
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
import {
  createAnalysisJob,
  getAnalysisJobStatus,
  getAnalysisResult,
  resolveAnalysisMediaUrl,
  type AnalysisResultResponse,
  type JobStatusResponse,
} from "@/lib/analysis-api";

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

type TimelineSegment = {
  label: string;
  width: string;
  accent: string;
  start: number;
  end: number;
};

type WorkspaceTrack = {
  id: string;
  label: string;
  instrumentKey: InstrumentKey | null;
  audioUrl: string | null;
  scoreUrl: string | null;
  midiUrl: string | null;
  noteSummary: string;
  source: string | null;
};

type LiveWorkspaceView = {
  archive: ArchiveItem;
  availableInstruments: InstrumentKey[];
  timeline: TimelineSegment[];
  tracks: WorkspaceTrack[];
  scoresByInstrument: Record<InstrumentKey, { title: string; scoreRows: ScoreRow[] }>;
  sourceTypeLabel: string;
  compareOriginalAudioUrl: string | null;
  compareResynthAudioUrl: string | null;
  compareSummary: string;
  noteCount: number;
  scoreUrl: string | null;
  midiUrl: string | null;
  effectMessage: string | null;
  evaluation: AnalysisResultResponse["evaluation"];
  warnings: string[];
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

const timeline: TimelineSegment[] = [
  { label: "인트로", width: "18%", accent: "bg-fuchsia-400/80", start: 0, end: 32 },
  { label: "벌스", width: "24%", accent: "bg-white/65", start: 32, end: 76 },
  { label: "프리코러스", width: "14%", accent: "bg-fuchsia-300/70", start: 76, end: 102 },
  { label: "코러스", width: "28%", accent: "bg-white/80", start: 102, end: 154 },
  { label: "브리지", width: "16%", accent: "bg-fuchsia-500/70", start: 154, end: 186 },
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

export type PageKey =
  | "home"
  | "archive"
  | "analysis"
  | "instruments"
  | "score"
  | "effects"
  | "compare";

type PageStat = {
  label: string;
  value: string;
};

const workspaceStorageKeys = {
  archiveId: "music-archive:selected-archive-id",
  instrument: "music-archive:selected-instrument",
  workspacePrepared: "music-archive:workspace-prepared",
} as const;

function getStoredArchiveId() {
  if (typeof window === "undefined") {
    return archiveItems[2].id;
  }

  const storedArchiveId = window.localStorage.getItem(workspaceStorageKeys.archiveId);
  if (storedArchiveId && archiveItems.some((item) => item.id === storedArchiveId)) {
    return storedArchiveId;
  }

  return archiveItems[2].id;
}

function getStoredInstrument(): InstrumentKey {
  if (typeof window === "undefined") {
    return "기타";
  }

  const storedInstrument = window.localStorage.getItem(workspaceStorageKeys.instrument);
  if (
    storedInstrument &&
    (Object.keys(instrumentContents) as InstrumentKey[]).includes(
      storedInstrument as InstrumentKey,
    )
  ) {
    return storedInstrument as InstrumentKey;
  }

  return "기타";
}

function getStoredWorkspacePrepared() {
  if (typeof window === "undefined") {
    return false;
  }

  return window.localStorage.getItem(workspaceStorageKeys.workspacePrepared) === "true";
}

const analysisStorageKeys = {
  jobId: "music-archive:analysis-job-id",
  sourceType: "music-archive:analysis-source-type",
} as const;

const revealTimelineAccents = [
  "bg-fuchsia-400/80",
  "bg-white/65",
  "bg-fuchsia-300/70",
  "bg-white/80",
  "bg-fuchsia-500/70",
  "bg-white/72",
] as const;

function getStoredAnalysisJobId() {
  if (typeof window === "undefined") {
    return null;
  }

  return window.localStorage.getItem(analysisStorageKeys.jobId);
}

function getStoredAnalysisSourceType() {
  if (typeof window === "undefined") {
    return "";
  }

  return window.localStorage.getItem(analysisStorageKeys.sourceType) ?? "";
}

function getSourceTypeLabel(sourceType?: string | null) {
  if (sourceType === "upload") {
    return "파일 업로드";
  }

  if (sourceType === "link") {
    return "링크 입력";
  }

  if (sourceType === "sample") {
    return "샘플 기반";
  }

  return "실시간 분석";
}

function formatDurationLabel(duration: number) {
  const totalSeconds = Math.max(0, Math.round(duration));
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

function formatAnalysisTime(seconds: number) {
  const safeSeconds = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(safeSeconds / 60);
  const remainingSeconds = safeSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remainingSeconds).padStart(2, "0")}`;
}

function clampMetric(value: number | null | undefined, fallback = 0) {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return fallback;
  }

  return Math.max(0, Math.min(1, value));
}

function mapTrackInstrument(instrument: string): InstrumentKey | null {
  const normalized = instrument.trim().toLowerCase();

  if (normalized.includes("보컬") || normalized.includes("vocal")) {
    return "보컬";
  }

  if (normalized.includes("베이스") || normalized.includes("bass")) {
    return "베이스";
  }

  if (normalized.includes("드럼") || normalized.includes("drum")) {
    return "드럼";
  }

  if (normalized.includes("피아노") || normalized.includes("piano")) {
    return "피아노";
  }

  if (normalized.includes("기타") || normalized.includes("other") || normalized.includes("guitar")) {
    return "기타";
  }

  return null;
}

function getRecommendedInstrumentFromResult(result: AnalysisResultResponse): InstrumentKey {
  const trackInstruments = result.tracks
    .map((track) => mapTrackInstrument(track.instrument))
    .filter((value): value is InstrumentKey => value !== null);

  const preferredOrder: InstrumentKey[] = ["기타", "보컬", "베이스", "드럼", "피아노", "기타(other)"];

  for (const item of preferredOrder) {
    if (trackInstruments.includes(item)) {
      return item;
    }
  }

  return "기타";
}

function describeLevel(value: number) {
  if (value >= 0.72) {
    return "뚜렷하게 감지됨";
  }

  if (value >= 0.42) {
    return "중간 이상으로 감지됨";
  }

  if (value >= 0.2) {
    return "약하게 감지됨";
  }

  return "거의 감지되지 않음";
}

function buildEffectMetrics(effectInfo: AnalysisResultResponse["guitar_effects"]): EffectMetric[] {
  return [
    {
      label: "드라이브",
      value: clampMetric(effectInfo.drive),
      detail: `드라이브 성향이 ${describeLevel(clampMetric(effectInfo.drive))}.`,
    },
    {
      label: "공간계",
      value: clampMetric(effectInfo.space),
      detail: `공간계 잔향이 ${describeLevel(clampMetric(effectInfo.space))}.`,
    },
    {
      label: "모듈레이션",
      value: clampMetric(effectInfo.modulation),
      detail: `모듈레이션 흔들림이 ${describeLevel(clampMetric(effectInfo.modulation))}.`,
    },
    {
      label: "톤 밝기",
      value: clampMetric(effectInfo.tone_brightness),
      detail: `톤 밝기가 ${describeLevel(clampMetric(effectInfo.tone_brightness))}.`,
    },
  ];
}

function buildTimelineFromSections(
  sections: AnalysisResultResponse["sections"],
  duration: number,
): TimelineSegment[] {
  if (sections.length === 0 || duration <= 0) {
    return timeline.map((item, index) => ({
      ...item,
      start: index * 10,
      end: (index + 1) * 10,
    }));
  }

  return sections.slice(0, 8).map((section, index) => {
    const width = Math.max(8, ((section.end - section.start) / duration) * 100);

    return {
      label: section.label,
      width: `${Math.min(width, 100)}%`,
      accent: revealTimelineAccents[index % revealTimelineAccents.length],
      start: section.start,
      end: section.end,
    };
  });
}

function buildHighlightsFromResult(
  result: AnalysisResultResponse,
  sourceTypeLabel: string,
) {
  const highlights: string[] = [];
  const [firstSection, secondSection] = result.sections;
  const availableTrackLabels = result.tracks
    .filter((track) => track.instrument !== "원본 믹스")
    .map((track) => track.instrument);

  if (firstSection) {
    const chordText =
      firstSection.chords.length > 0
        ? firstSection.chords.slice(0, 3).join(" → ")
        : "코드 정보가 아직 정리되지 않았습니다.";
    highlights.push(
      `${firstSection.label} (${formatAnalysisTime(firstSection.start)}-${formatAnalysisTime(firstSection.end)}) 구간은 ${chordText} 흐름을 중심으로 정리됩니다.`,
    );
  }

  if (secondSection) {
    const detailParts = [
      secondSection.dynamics ? `다이내믹 ${secondSection.dynamics}` : null,
      secondSection.effects[0] ?? null,
      secondSection.instruments[0] ? `${secondSection.instruments[0]} 중심` : null,
    ].filter(Boolean);
    highlights.push(
      `${secondSection.label} 구간은 ${detailParts.join(" · ") || "구간 특성 정보"}로 읽힙니다.`,
    );
  }

  if (availableTrackLabels.length > 0) {
    highlights.push(
      `${sourceTypeLabel} 기준으로 ${availableTrackLabels.join(", ")} stem이 분리되어 바로 들어볼 수 있습니다.`,
    );
  }

  if (result.guitar_effects.labels.length > 0) {
    highlights.push(
      `이펙터 추정은 ${result.guitar_effects.labels.join(", ")} 성향을 중심으로 감지되었습니다.`,
    );
  }

  return highlights.slice(0, 4);
}

function buildTrackRows(
  result: AnalysisResultResponse,
  instrument: InstrumentKey,
): ScoreRow[] {
  if (result.sections.length === 0) {
    return instrumentContents[instrument].scoreRows;
  }

  return result.sections.slice(0, 6).map((section, index) => {
    const cueParts = [
      section.dynamics ? `다이내믹 ${section.dynamics}` : null,
      section.effects[0] ?? null,
      section.source ?? null,
    ].filter(Boolean);
    const noteSummary =
      instrument === "보컬" || instrument === "기타"
        ? `멜로디/표기 확인 구간 ${formatAnalysisTime(section.start)}-${formatAnalysisTime(section.end)}`
        : `${instrument} 역할 확인 구간 ${formatAnalysisTime(section.start)}-${formatAnalysisTime(section.end)}`;

    return {
      bar: String(index + 1).padStart(2, "0"),
      chord: section.chords[0] ?? "-",
      note: `${section.label} · ${noteSummary}`,
      cue: cueParts.join(" · ") || "구간 설명 없음",
    };
  });
}

function buildScoreContentsFromResult(result: AnalysisResultResponse) {
  return (Object.keys(instrumentContents) as InstrumentKey[]).reduce(
    (accumulator, instrument) => {
      accumulator[instrument] = {
        title: `${instrument} 분석 노트`,
        scoreRows: buildTrackRows(result, instrument),
      };
      return accumulator;
    },
    {} as Record<InstrumentKey, { title: string; scoreRows: ScoreRow[] }>,
  );
}

function buildWorkspaceTracks(result: AnalysisResultResponse): WorkspaceTrack[] {
  return result.tracks.map((track, index) => ({
    id: `${track.instrument}-${index}`,
    label: track.instrument,
    instrumentKey: mapTrackInstrument(track.instrument),
    audioUrl: resolveAnalysisMediaUrl(track.audio_url),
    scoreUrl: resolveAnalysisMediaUrl(track.score_url),
    midiUrl: resolveAnalysisMediaUrl(track.midi_url),
    noteSummary: track.note_summary || "설명 없음",
    source: track.source,
  }));
}

function buildLiveWorkspaceView(
  result: AnalysisResultResponse,
  fallbackArchive: ArchiveItem,
  sourceType: string,
): LiveWorkspaceView {
  const sourceTypeLabel = getSourceTypeLabel(sourceType);
  const recommendedInstrument = getRecommendedInstrumentFromResult(result);
  const tracks = buildWorkspaceTracks(result);
  const availableInstruments = Array.from(
    new Set(
      tracks
        .map((track) => track.instrumentKey)
        .filter((value): value is InstrumentKey => value !== null),
    ),
  );
  const similarity = clampMetric(result.evaluation.similarity, fallbackArchive.similarity);
  const effectMetrics = buildEffectMetrics(result.guitar_effects);
  const timelineItems = buildTimelineFromSections(result.sections, result.meta.duration);
  const highlights = buildHighlightsFromResult(result, sourceTypeLabel);
  const scoreAssetTrack =
    tracks.find((track) => track.instrumentKey === recommendedInstrument && track.scoreUrl) ||
    tracks.find((track) => track.scoreUrl) ||
    null;

  return {
    archive: {
      ...fallbackArchive,
      title: result.meta.title || fallbackArchive.title,
      artist: sourceTypeLabel,
      genre: `${sourceTypeLabel} · 실시간 분석`,
      focus:
        availableInstruments.length > 0
          ? `${availableInstruments.join(" · ")} 파트 분리 결과`
          : fallbackArchive.focus,
      bpm: Math.round(result.meta.bpm ?? fallbackArchive.bpm),
      key: result.meta.key || fallbackArchive.key,
      similarity,
      summary:
        result.meta.analysis_summary ||
        `${sourceTypeLabel}로 들어온 음원을 02ver 백엔드 파이프라인으로 분석한 결과입니다.`,
      durationLabel: formatDurationLabel(result.meta.duration || 0),
      recommendedInstrument,
      moodTags:
        result.guitar_effects.labels.length > 0
          ? result.guitar_effects.labels
          : [sourceTypeLabel, "Live", "Analysis"],
      highlights:
        highlights.length > 0 ? highlights : fallbackArchive.highlights,
      effectMetrics,
    },
    availableInstruments:
      availableInstruments.length > 0
        ? availableInstruments
        : [recommendedInstrument],
    timeline: timelineItems,
    tracks,
    scoresByInstrument: buildScoreContentsFromResult(result),
    sourceTypeLabel,
    compareOriginalAudioUrl: resolveAnalysisMediaUrl(result.compare.original_audio_url),
    compareResynthAudioUrl: resolveAnalysisMediaUrl(result.compare.resynth_audio_url),
    compareSummary:
      result.evaluation.similarity !== null
        ? `전체 유사도 ${(result.evaluation.similarity * 100).toFixed(1)}% 기준으로 원음원과 재합성 결과를 비교합니다.`
        : "원음원과 재합성 결과를 같은 화면에서 비교합니다.",
    noteCount: result.melody_notes.length,
    scoreUrl: scoreAssetTrack?.scoreUrl ?? null,
    midiUrl: scoreAssetTrack?.midiUrl ?? null,
    effectMessage: result.guitar_effects.message,
    evaluation: result.evaluation,
    warnings: result.warnings,
  };
}

const primaryNav: { key: Exclude<PageKey, "home">; label: string; href: string }[] = [
  { key: "archive", label: "아카이브", href: "/archive" },
  { key: "analysis", label: "분석 기능", href: "/analysis" },
  { key: "instruments", label: "악기별 탐색", href: "/instruments" },
  { key: "score", label: "악보", href: "/score" },
  { key: "effects", label: "이펙터", href: "/effects" },
  { key: "compare", label: "비교분석", href: "/compare" },
];

const pageRouteSequence: { key: PageKey; href: string }[] = [
  { key: "home", href: "/" },
  ...primaryNav,
];

const workspacePages: { key: Extract<PageKey, "instruments" | "score" | "effects" | "compare">; label: string; href: string; description: string }[] = [
  {
    key: "instruments",
    label: "악기별 탐색",
    href: "/instruments",
    description: "파트를 바꿔가며 구조와 역할을 읽는 페이지",
  },
  {
    key: "score",
    label: "악보",
    href: "/score",
    description: "표기와 구간 주석을 집중해서 보는 페이지",
  },
  {
    key: "effects",
    label: "이펙터",
    href: "/effects",
    description: "톤과 공간계 프로파일을 읽는 페이지",
  },
  {
    key: "compare",
    label: "비교분석",
    href: "/compare",
    description: "원음원과 재합성 결과를 비교하는 페이지",
  },
];

function useRevealOnce<T extends HTMLElement>() {
  const ref = useRef<T | null>(null);
  const [isVisible, setIsVisible] = useState(
    () =>
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches,
  );

  useEffect(() => {
    const element = ref.current;

    if (!element || isVisible) {
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const [entry] = entries;

        if (!entry?.isIntersecting) {
          return;
        }

        setIsVisible(true);
        observer.disconnect();
      },
      {
        threshold: 0.18,
        rootMargin: "0px 0px -12% 0px",
      },
    );

    observer.observe(element);

    return () => {
      observer.disconnect();
    };
  }, [isVisible]);

  return { ref, isVisible };
}

function getRevealItemStyle(index: number, baseDelay = 110): CSSProperties {
  return {
    transitionDelay: `${baseDelay + index * 85}ms`,
  };
}

function isInteractiveWheelTarget(target: EventTarget | null) {
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  return Boolean(
    target.closest(
      'input, textarea, select, button, a, summary, [role="dialog"], [contenteditable="true"]',
    ),
  );
}

function RevealItem({
  children,
  className = "",
  index,
  isVisible,
  hoverLift = false,
}: {
  children: ReactNode;
  className?: string;
  index: number;
  isVisible: boolean;
  hoverLift?: boolean;
}) {
  return (
    <div
      className={`motion-safe:transform-gpu motion-safe:will-change-transform transition-[opacity,transform] duration-[720ms] ease-[cubic-bezier(0.22,1,0.36,1)] ${
        isVisible ? "translate-y-0 opacity-100" : "translate-y-7 opacity-0"
      } ${hoverLift ? "motion-safe:hover:-translate-y-1.5" : ""} ${className}`}
      style={getRevealItemStyle(index)}
    >
      {children}
    </div>
  );
}

export function MusicArchiveExperience({ page }: { page: PageKey }) {
  const router = useRouter();
  const headerRef = useRef<HTMLElement | null>(null);
  const lastHorizontalNavigationAtRef = useRef(0);
  const [selectedArchiveId, setSelectedArchiveId] = useState(archiveItems[2].id);
  const [hoveredArchiveId, setHoveredArchiveId] = useState<string | null>(null);
  const [archiveSpotlightId, setArchiveSpotlightId] = useState(archiveItems[0].id);
  const [leadViewportHeight, setLeadViewportHeight] = useState(0);
  const [selectedInstrument, setSelectedInstrument] = useState<InstrumentKey>("기타");
  const [requestDialogOpen, setRequestDialogOpen] = useState(false);
  const [requestDraft, setRequestDraft] = useState<RequestDraft>(() =>
    createRequestDraft(archiveItems[2].id),
  );
  const [requestStatus, setRequestStatus] = useState<RequestStatus>("editing");
  const [requestStageIndex, setRequestStageIndex] = useState(0);
  const [requestError, setRequestError] = useState("");
  const [workspacePrepared, setWorkspacePrepared] = useState(false);
  const [analysisJobId, setAnalysisJobId] = useState<string | null>(null);
  const [analysisSourceType, setAnalysisSourceType] = useState("");
  const [analysisJobStatus, setAnalysisJobStatus] = useState<JobStatusResponse | null>(null);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResultResponse | null>(null);
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
  const requestArchive = useMemo(
    () => getArchiveById(requestDraft.archiveId),
    [requestDraft.archiveId],
  );
  const previewArchive = useMemo(
    () => (samplePreviewId ? getArchiveById(samplePreviewId) : null),
    [samplePreviewId],
  );
  const archiveSpotlightItem = useMemo(
    () => getArchiveById(archiveSpotlightId),
    [archiveSpotlightId],
  );
  const liveWorkspace = useMemo(
    () =>
      analysisResult
        ? buildLiveWorkspaceView(analysisResult, selectedArchive, analysisSourceType)
        : null,
    [analysisResult, analysisSourceType, selectedArchive],
  );
  const workspaceArchive = liveWorkspace?.archive ?? selectedArchive;
  const availableInstruments = liveWorkspace?.availableInstruments ??
    (Object.keys(instrumentContents) as InstrumentKey[]);
  const activeTimeline = liveWorkspace?.timeline ?? timeline;
  const workspaceTracks = liveWorkspace?.tracks ?? [];
  const activeInstrument = availableInstruments.includes(selectedInstrument)
    ? selectedInstrument
    : liveWorkspace?.archive.recommendedInstrument ?? selectedInstrument;
  const selectedScore = (liveWorkspace?.scoresByInstrument ?? instrumentContents)[activeInstrument];
  const activeCompareSection = activeTimeline.some(
    (item) => item.label === comparePreview.activeSection,
  )
    ? comparePreview.activeSection
    : activeTimeline[0]?.label ?? comparePreview.activeSection;
  const layoutViewportStyle = useMemo<CSSProperties>(
    () => ({
      ["--lead-viewport-height" as string]: leadViewportHeight
        ? `${leadViewportHeight}px`
        : "calc(100dvh - 88px)",
    }),
    [leadViewportHeight],
  );

  function clearLiveWorkspace() {
    setAnalysisJobId(null);
    setAnalysisSourceType("");
    setAnalysisJobStatus(null);
    setAnalysisResult(null);
  }

  useEffect(() => {
    const timeoutId = window.setTimeout(() => {
      setSelectedArchiveId(getStoredArchiveId());
      setSelectedInstrument(getStoredInstrument());
      setWorkspacePrepared(getStoredWorkspacePrepared());
      setAnalysisJobId(getStoredAnalysisJobId());
      setAnalysisSourceType(getStoredAnalysisSourceType());
    }, 0);

    return () => {
      window.clearTimeout(timeoutId);
    };
  }, []);

  useEffect(() => {
    const updateLeadViewportHeight = () => {
      const viewportHeight = window.visualViewport?.height ?? window.innerHeight;
      const headerHeight = headerRef.current?.getBoundingClientRect().height ?? 0;
      setLeadViewportHeight(Math.max(Math.round(viewportHeight - headerHeight), 560));
    };

    updateLeadViewportHeight();

    const headerObserver =
      typeof ResizeObserver === "undefined"
        ? null
        : new ResizeObserver(updateLeadViewportHeight);

    if (headerRef.current && headerObserver) {
      headerObserver.observe(headerRef.current);
    }

    const visualViewport = window.visualViewport;
    window.addEventListener("resize", updateLeadViewportHeight);
    visualViewport?.addEventListener("resize", updateLeadViewportHeight);

    return () => {
      window.removeEventListener("resize", updateLeadViewportHeight);
      visualViewport?.removeEventListener("resize", updateLeadViewportHeight);
      headerObserver?.disconnect();
    };
  }, []);

  useEffect(() => {
    window.localStorage.setItem(workspaceStorageKeys.archiveId, selectedArchiveId);
  }, [selectedArchiveId]);

  useEffect(() => {
    window.localStorage.setItem(workspaceStorageKeys.instrument, selectedInstrument);
  }, [selectedInstrument]);

  useEffect(() => {
    window.localStorage.setItem(
      workspaceStorageKeys.workspacePrepared,
      String(workspacePrepared),
    );
  }, [workspacePrepared]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    if (!analysisJobId) {
      window.localStorage.removeItem(analysisStorageKeys.jobId);
      window.localStorage.removeItem(analysisStorageKeys.sourceType);
      return;
    }

    window.localStorage.setItem(analysisStorageKeys.jobId, analysisJobId);
    if (analysisSourceType) {
      window.localStorage.setItem(analysisStorageKeys.sourceType, analysisSourceType);
    }
  }, [analysisJobId, analysisSourceType]);

  useEffect(() => {
    if (requestStatus !== "preparing") {
      return;
    }

    if (requestDraft.sourceMode !== "sample") {
      return;
    }

    const timeoutId = window.setTimeout(() => {
      if (requestStageIndex >= requestStages.length - 1) {
        const preparedArchive = getArchiveById(requestDraft.archiveId);
        setAnalysisJobId(null);
        setAnalysisSourceType("");
        setAnalysisJobStatus(null);
        setAnalysisResult(null);
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
  }, [requestDraft.archiveId, requestDraft.sourceMode, requestStageIndex, requestStatus]);

  useEffect(() => {
    if (!analysisJobId) {
      return;
    }

    let cancelled = false;
    let timeoutId = 0;

    const pollStatus = async () => {
      try {
        const nextStatus = await getAnalysisJobStatus(analysisJobId);

        if (cancelled) {
          return;
        }

        setAnalysisJobStatus(nextStatus);
        setAnalysisSourceType(nextStatus.source_type);

        if (nextStatus.status === "failed") {
          setRequestError(nextStatus.error_message || "분석 파이프라인 실행에 실패했습니다.");
          setRequestStatus("editing");
          return;
        }

        if (nextStatus.status === "completed" && nextStatus.result_available) {
          const nextResult = await getAnalysisResult(analysisJobId);

          if (cancelled) {
            return;
          }

          setAnalysisResult(nextResult);
          setWorkspacePrepared(true);
          setRequestStatus("ready");
          setRequestError("");
          setSelectedInstrument(getRecommendedInstrumentFromResult(nextResult));
          return;
        }

        timeoutId = window.setTimeout(pollStatus, 1800);
      } catch (error) {
        if (cancelled) {
          return;
        }

        clearLiveWorkspace();
        setRequestError(
          error instanceof Error
            ? error.message
            : "백엔드 연결 상태를 다시 확인해 주세요.",
        );
        setRequestStatus("editing");
      }
    };

    void pollStatus();

    return () => {
      cancelled = true;
      window.clearTimeout(timeoutId);
    };
  }, [analysisJobId]);

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

  useEffect(() => {
    const currentPageIndex = pageRouteSequence.findIndex((item) => item.key === page);

    if (currentPageIndex === -1) {
      return;
    }

    const handleWheel = (event: WheelEvent) => {
      if (requestDialogOpen || samplePreviewId || comparePreview.open) {
        return;
      }

      if (isInteractiveWheelTarget(event.target)) {
        return;
      }

      const horizontalDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY) * 1.15
        ? event.deltaX
        : event.shiftKey
          ? event.deltaY
          : 0;

      if (Math.abs(horizontalDelta) < 48) {
        return;
      }

      const now = Date.now();
      if (now - lastHorizontalNavigationAtRef.current < 850) {
        return;
      }

      const nextIndex =
        horizontalDelta > 0 ? currentPageIndex + 1 : currentPageIndex - 1;
      const targetRoute = pageRouteSequence[nextIndex];

      if (!targetRoute) {
        return;
      }

      lastHorizontalNavigationAtRef.current = now;
      router.push(targetRoute.href);
    };

    window.addEventListener("wheel", handleWheel, { passive: true });

    return () => {
      window.removeEventListener("wheel", handleWheel);
    };
  }, [comparePreview.open, page, requestDialogOpen, router, samplePreviewId]);

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
    if (requestStatus !== "preparing") {
      resetRequestDialogState();
    }
  };

  const handleRequestSubmit = async (event: FormEvent<HTMLFormElement>) => {
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

    if (requestDraft.sourceMode === "sample") {
      return;
    }

    try {
      const createdJob = await createAnalysisJob({
        file: requestDraft.sourceMode === "upload" ? requestDraft.file : null,
        sourceLink: requestDraft.sourceMode === "link" ? requestDraft.sourceLink : "",
        title: getRequestTitle(requestDraft),
      });

      setAnalysisJobId(createdJob.job_id);
      setAnalysisSourceType(createdJob.source_type);
      setAnalysisJobStatus(createdJob);
    } catch (error) {
      clearLiveWorkspace();
      setRequestStatus("editing");
      setRequestError(
        error instanceof Error
          ? error.message
          : "분석 요청을 시작하지 못했습니다.",
      );
    }
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
    clearLiveWorkspace();
    setSelectedArchiveId(targetArchive.id);
    setArchiveSpotlightId(targetArchive.id);
    setSelectedInstrument(targetArchive.recommendedInstrument);
    setWorkspacePrepared(true);
    setSamplePreviewId(null);
    router.push("/instruments");
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

  let pageTitle = "";
  let pageDescription = "";
  let pageStats: PageStat[] = [];
  let pageActions: ReactNode = null;
  let pageContent: ReactNode = null;

  if (page === "home") {
    pageContent = (
      <>
        <Hero
          item={workspaceArchive}
          selectedInstrument={activeInstrument}
          workspacePrepared={workspacePrepared}
          onSampleClick={() => router.push("/archive")}
          onUploadClick={() => openRequestDialog("upload")}
        />
        <HomePathwaysSection />
        <ArchiveSection
          hoveredArchiveId={hoveredArchiveId}
          selectedArchiveId={selectedArchiveId}
          onHoverArchive={setHoveredArchiveId}
          onInspectArchive={setArchiveSpotlightId}
          onPreviewArchive={setSamplePreviewId}
        />
        <FeatureSection />
      </>
    );
  } else if (page === "archive") {
    pageTitle = "분석 가능한 샘플 아카이브";
    pageDescription =
      "샘플을 고르는 페이지입니다. 여기서는 곡의 무드와 분석 포인트를 보고, 어떤 곡으로 워크스페이스를 열지 결정합니다.";
    pageStats = [
      { label: "Sample", value: `${archiveItems.length} tracks` },
      { label: "Selected", value: archiveSpotlightItem.title },
      { label: "Focus", value: archiveSpotlightItem.recommendedInstrument },
    ];
    pageActions = null;
    pageContent = (
      <>
        <PageLead
          eyebrow="Archive"
          title={pageTitle}
          description={pageDescription}
          stats={pageStats}
          actions={pageActions}
        />
        <ArchiveSection
          hoveredArchiveId={hoveredArchiveId}
          selectedArchiveId={selectedArchiveId}
          onHoverArchive={setHoveredArchiveId}
          onInspectArchive={setArchiveSpotlightId}
          onPreviewArchive={setSamplePreviewId}
        />
        <ArchiveContextSection item={archiveSpotlightItem} />
      </>
    );
  } else if (page === "analysis") {
    pageTitle = "분석 기능을 한눈에 보는 페이지";
    pageDescription =
      "분석 기능 페이지는 어떤 결과를 만들 수 있는지와 요청 창이 어떤 출력을 준비하는지를 설명하는 역할을 맡습니다.";
    pageStats = [
      { label: "Outputs", value: `${analysisOptions.length} modules` },
      { label: "Anchor", value: workspaceArchive.title },
      { label: "Pipeline", value: `${requestStages.length} steps` },
    ];
    pageActions = null;
    pageContent = (
      <>
        <PageLead
          eyebrow="Analysis"
          title={pageTitle}
          description={pageDescription}
          stats={pageStats}
          actions={pageActions}
        />
        <FeatureSection />
        <AnalysisWorkflowSection onOpenRequest={() => openRequestDialog("sample")} />
      </>
    );
  } else if (page === "instruments") {
    pageTitle = "악기별 탐색 워크스페이스";
    pageDescription =
      "악기별 탐색 페이지는 파트 전환과 역할 해석에 집중합니다. 어떤 악기가 곡의 중심을 만드는지 읽는 데 초점을 둡니다.";
    pageStats = [
      { label: "Track", value: workspaceArchive.title },
      { label: "Active Part", value: activeInstrument },
      { label: "Source", value: liveWorkspace?.sourceTypeLabel ?? workspaceArchive.artist },
    ];
    pageActions = null;
    pageContent = (
      <>
        <PageLead
          eyebrow="Instrument Explorer"
          title={pageTitle}
          description={pageDescription}
          stats={pageStats}
          actions={pageActions}
        />
        <WorkspacePageTabs activePage={page} />
        <AnalysisShowcase
          selectedArchive={workspaceArchive}
          selectedInstrument={activeInstrument}
          onSelectInstrument={setSelectedInstrument}
          selectedScore={selectedScore}
          availableInstruments={availableInstruments}
          timeline={activeTimeline}
          workspaceTracks={workspaceTracks}
          liveSummary={liveWorkspace?.sourceTypeLabel ?? ""}
        />
      </>
    );
  } else if (page === "score") {
    pageTitle = "악보 해석과 구간 주석 페이지";
    pageDescription =
      "악보 페이지는 표기, 구간 타임라인, 코드 해석을 중심으로 구성됩니다. 악기별 탐색보다 notation 자체에 더 집중합니다.";
    pageStats = [
      { label: "Track", value: workspaceArchive.title },
      { label: "Notation", value: activeInstrument },
      { label: "Timeline", value: workspaceArchive.durationLabel },
    ];
    pageActions = null;
    pageContent = (
      <>
        <PageLead
          eyebrow="Score Workspace"
          title={pageTitle}
          description={pageDescription}
          stats={pageStats}
          actions={pageActions}
        />
        <WorkspacePageTabs activePage={page} />
        <ScoreFocusSection
          selectedArchive={workspaceArchive}
          selectedInstrument={activeInstrument}
          onSelectInstrument={setSelectedInstrument}
          selectedScore={selectedScore}
          availableInstruments={availableInstruments}
          timeline={activeTimeline}
          noteCount={liveWorkspace?.noteCount ?? 0}
          scoreUrl={liveWorkspace?.scoreUrl ?? null}
          midiUrl={liveWorkspace?.midiUrl ?? null}
          warnings={liveWorkspace?.warnings ?? []}
        />
      </>
    );
  } else if (page === "effects") {
    pageTitle = "톤과 이펙터를 읽는 페이지";
    pageDescription =
      "이펙터 페이지는 드라이브, 공간계, 모듈레이션, 톤 밝기처럼 소리의 질감과 성향을 집중적으로 읽는 역할을 맡습니다.";
    pageStats = [
      { label: "Track", value: workspaceArchive.title },
      { label: "Profile", value: activeInstrument },
      { label: "Similarity", value: workspaceArchive.similarity.toFixed(2) },
    ];
    pageActions = null;
    pageContent = (
      <>
        <PageLead
          eyebrow="Effects Lab"
          title={pageTitle}
          description={pageDescription}
          stats={pageStats}
          actions={pageActions}
        />
        <WorkspacePageTabs activePage={page} />
        <EffectsLabSection
          selectedArchive={workspaceArchive}
          selectedInstrument={activeInstrument}
          onSelectInstrument={setSelectedInstrument}
          availableInstruments={availableInstruments}
          timeline={activeTimeline}
          effectMessage={liveWorkspace?.effectMessage ?? null}
        />
      </>
    );
  } else {
    pageTitle = "원음원과 재합성을 비교하는 페이지";
    pageDescription =
      "비교분석 페이지는 A/B 청취 흐름과 정량 지표를 확인하는 역할입니다. 다른 페이지에서 읽은 내용을 마지막에 검증하는 곳입니다.";
    pageStats = [
      { label: "Track", value: workspaceArchive.title },
      { label: "Similarity", value: workspaceArchive.similarity.toFixed(2) },
      { label: "Compare", value: "A/B Ready" },
    ];
    pageActions = null;
    pageContent = (
      <>
        <PageLead
          eyebrow="Compare"
          title={pageTitle}
          description={pageDescription}
          stats={pageStats}
          actions={pageActions}
        />
        <WorkspacePageTabs activePage={page} />
        <CompareSection
          comparePreview={comparePreview}
          similarity={workspaceArchive.similarity}
          onToggleCard={handleCompareCardToggle}
          originalAudioUrl={liveWorkspace?.compareOriginalAudioUrl ?? null}
          resynthAudioUrl={liveWorkspace?.compareResynthAudioUrl ?? null}
          evaluation={liveWorkspace?.evaluation ?? null}
          compareSummary={liveWorkspace?.compareSummary ?? ""}
        />
      </>
    );
  }

  return (
    <>
      <main
        className="min-h-screen bg-[#0d0f12] text-white"
        style={layoutViewportStyle}
      >
        <Header
          activePage={page}
          headerRef={headerRef}
          onStart={() => openRequestDialog("sample")}
        />
        {pageContent}
        <Footer onStart={() => openRequestDialog("sample")} />
      </main>

      {requestDialogOpen ? (
        <AnalysisRequestDialog
          draft={requestDraft}
          errorMessage={requestError}
          jobStatus={analysisJobStatus}
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
            router.push("/instruments");
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
          activeSection={activeCompareSection}
          isPlaying={comparePreview.isPlaying}
          progress={comparePreview.progress}
          selectedArchive={workspaceArchive}
          timeline={activeTimeline}
          originalAudioUrl={liveWorkspace?.compareOriginalAudioUrl ?? null}
          resynthAudioUrl={liveWorkspace?.compareResynthAudioUrl ?? null}
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

export default MusicArchiveExperience;

function Header({
  activePage,
  headerRef,
  onStart,
}: {
  activePage: PageKey;
  headerRef: RefObject<HTMLElement | null>;
  onStart: () => void;
}) {
  return (
    <header
      ref={headerRef}
      className="sticky top-0 z-50 border-b border-white/10 bg-[#0d0f12]/80 backdrop-blur-xl"
    >
      <div className="mx-auto flex max-w-[1600px] items-center justify-between px-8 py-5">
        <Link
          href="/"
          className="flex items-center gap-3 text-lg font-semibold tracking-[0.28em] text-white/92"
        >
          <Disc3 className="h-5 w-5 text-fuchsia-300" />
          <span>MUSIC ARCHIVE</span>
        </Link>

        <nav className="hidden items-center gap-8 text-[15px] text-white/72 lg:flex">
          {primaryNav.map((item) => (
            <NavItem
              key={item.key}
              href={item.href}
              isActive={activePage === item.key}
              label={item.label}
            />
          ))}
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

function NavItem({
  label,
  href,
  isActive,
}: {
  label: string;
  href: string;
  isActive: boolean;
}) {
  return (
    <Link
      href={href}
      className={`group relative transition hover:text-white ${
        isActive ? "text-white" : ""
      }`}
    >
      {label}
      <span
        className={`absolute -bottom-2 left-0 h-px bg-fuchsia-300 transition-all duration-300 ${
          isActive ? "w-full" : "w-0 group-hover:w-full"
        }`}
      />
    </Link>
  );
}

function PageLead({
  eyebrow,
  title,
  description,
  stats,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  stats: PageStat[];
  actions: ReactNode;
}) {
  const { ref: statsRef, isVisible: areStatsVisible } = useRevealOnce<HTMLDivElement>();

  return (
    <section className="relative overflow-hidden border-b border-white/10">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(217,70,239,0.12),transparent_28%),radial-gradient(circle_at_bottom_right,rgba(255,255,255,0.05),transparent_28%)]" />
      <div className="relative mx-auto flex min-h-[var(--lead-viewport-height)] max-w-[1600px] items-center px-8 py-16 sm:py-20">
        <div className="grid w-full gap-10 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="max-w-[880px]">
            <p className="text-sm uppercase tracking-[0.35em] text-white/42">
              {eyebrow}
            </p>
            <h1 className="mt-6 text-5xl font-semibold leading-[0.96] tracking-[-0.05em] text-white sm:text-6xl lg:text-[88px]">
              {title}
            </h1>
            <p className="mt-8 max-w-[700px] text-lg leading-8 text-white/66 sm:text-xl">
              {description}
            </p>
            {actions ? (
              <div className="mt-10 flex flex-wrap gap-4">{actions}</div>
            ) : null}
          </div>

          <div ref={statsRef} className="grid gap-4">
            {stats.map((item, index) => (
              <RevealItem key={item.label} hoverLift index={index} isVisible={areStatsVisible}>
                <div className="rounded-[28px] border border-white/10 bg-white/[0.04] px-6 py-5">
                  <p className="text-xs uppercase tracking-[0.28em] text-white/40">
                    {item.label}
                  </p>
                  <p className="mt-3 text-2xl font-medium tracking-[-0.03em] text-white">
                    {item.value}
                  </p>
                </div>
              </RevealItem>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}

function HomePathwaysSection() {
  const { ref: pathwayRef, isVisible: arePathwaysVisible } =
    useRevealOnce<HTMLDivElement>();
  const pathwayCards = [
    {
      href: "/archive",
      eyebrow: "Archive",
      title: "샘플을 고르는 페이지",
      description: "곡의 무드와 분석 포인트를 먼저 보고 워크스페이스를 시작합니다.",
    },
    {
      href: "/analysis",
      eyebrow: "Analysis",
      title: "분석 기능을 설명하는 페이지",
      description: "무엇을 뽑아낼지와 어떤 출력을 준비하는지 한눈에 정리합니다.",
    },
    {
      href: "/instruments",
      eyebrow: "Workspace",
      title: "악기별 탐색 페이지",
      description: "파트를 바꿔가며 구조와 역할을 읽는 중심 화면입니다.",
    },
    {
      href: "/score",
      eyebrow: "Notation",
      title: "악보와 이펙터 페이지",
      description: "표기와 질감을 각각 분리해 역할이 다른 화면으로 보게 만듭니다.",
    },
  ];

  return (
    <section className="mx-auto max-w-[1600px] px-8 py-24">
      <div className="mb-12">
        <p className="text-sm uppercase tracking-[0.32em] text-white/42">Page Roles</p>
        <h2 className="mt-4 text-3xl font-semibold tracking-[-0.03em] text-white sm:text-5xl">
          이제 각 메뉴는
          <br />
          실제로 다른 페이지를 가리킵니다
        </h2>
      </div>

      <div ref={pathwayRef} className="grid gap-6 lg:grid-cols-4">
        {pathwayCards.map((item, index) => (
          <RevealItem key={item.href} hoverLift index={index} isVisible={arePathwaysVisible}>
            <Link
              href={item.href}
              className="block rounded-[28px] border border-white/10 bg-white/[0.04] p-7 transition hover:border-fuchsia-300/30 hover:bg-white/[0.06]"
            >
              <p className="text-xs uppercase tracking-[0.28em] text-white/40">
                {item.eyebrow}
              </p>
              <h3 className="mt-5 text-2xl font-medium text-white">{item.title}</h3>
              <p className="mt-4 text-sm leading-7 text-white/60">{item.description}</p>
            </Link>
          </RevealItem>
        ))}
      </div>
    </section>
  );
}

function ArchiveContextSection({ item }: { item: ArchiveItem }) {
  const { ref: contextRef, isVisible: isContextVisible } =
    useRevealOnce<HTMLDivElement>();

  return (
    <section className="border-t border-white/10 bg-[#12151a]">
      <div
        ref={contextRef}
        className="mx-auto grid max-w-[1600px] items-start gap-6 px-8 py-24 lg:grid-cols-[1.08fr_0.92fr]"
      >
        <RevealItem hoverLift index={0} isVisible={isContextVisible}>
          <div className="rounded-[30px] border border-white/10 bg-white/[0.04] p-7">
            <p className="text-xs uppercase tracking-[0.28em] text-white/40">
              Selected Sample
            </p>
            <h3 className="mt-4 text-3xl font-medium text-white">{item.title}</h3>
            <p className="mt-3 text-base text-white/62">
              {item.artist} · {item.genre}
            </p>
            <p className="mt-6 text-sm leading-7 text-white/58">{item.summary}</p>
            <div className="mt-8 flex flex-wrap gap-2">
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
        </RevealItem>

        <div className="grid max-w-[520px] gap-3 justify-self-end sm:grid-cols-3">
          {[
            { label: "추천 시작점", value: item.recommendedInstrument },
            { label: "재생 길이", value: item.durationLabel },
            { label: "유사도", value: item.similarity.toFixed(2) },
          ].map((detail, index) => (
            <RevealItem
              key={detail.label}
              hoverLift
              index={index + 1}
              isVisible={isContextVisible}
            >
              <DetailTile label={detail.label} value={detail.value} />
            </RevealItem>
          ))}
        </div>
      </div>
    </section>
  );
}

function AnalysisWorkflowSection({ onOpenRequest }: { onOpenRequest: () => void }) {
  const { ref: workflowRef, isVisible: isWorkflowVisible } =
    useRevealOnce<HTMLDivElement>();

  return (
    <section className="mx-auto max-w-[1600px] px-8 py-24">
      <div ref={workflowRef} className="grid gap-8 lg:grid-cols-[0.9fr_1.1fr]">
        <RevealItem hoverLift index={0} isVisible={isWorkflowVisible}>
          <div className="rounded-[30px] border border-white/10 bg-white/[0.04] p-7">
            <p className="text-xs uppercase tracking-[0.28em] text-white/40">Workflow</p>
            <h3 className="mt-4 text-3xl font-medium text-white">요청 창 이후의 역할</h3>
            <div className="mt-8 space-y-4">
              {requestStages.map((stage, index) => (
                <div
                  key={stage.title}
                  className="rounded-[22px] border border-white/10 bg-[#0d0f12] px-5 py-4"
                >
                  <p className="text-sm font-medium text-white">
                    {index + 1}. {stage.title}
                  </p>
                  <p className="mt-2 text-sm leading-6 text-white/54">{stage.detail}</p>
                </div>
              ))}
            </div>
            <button
              type="button"
              onClick={onOpenRequest}
              className="mt-8 rounded-full bg-white px-6 py-3 text-sm font-medium text-[#0f1113] transition hover:bg-white/92"
            >
              분석 요청 창 다시 열기
            </button>
          </div>
        </RevealItem>

        <div className="grid gap-6 sm:grid-cols-2">
          {features.map((feature, index) => {
            const Icon = feature.icon;

            return (
              <RevealItem
                key={feature.title}
                hoverLift
                index={index + 1}
                isVisible={isWorkflowVisible}
              >
                <div className="rounded-[28px] border border-white/10 bg-white/[0.04] p-7">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-fuchsia-300/10 text-fuchsia-200">
                    <Icon className="h-5 w-5" />
                  </div>
                  <h4 className="mt-6 text-2xl font-medium text-white">{feature.title}</h4>
                  <p className="mt-4 text-sm leading-7 text-white/60">
                    {feature.description}
                  </p>
                </div>
              </RevealItem>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function WorkspacePageTabs({
  activePage,
}: {
  activePage: Extract<PageKey, "instruments" | "score" | "effects" | "compare">;
}) {
  const { ref: tabsRef, isVisible: areTabsVisible } = useRevealOnce<HTMLDivElement>();

  return (
    <section className="mx-auto max-w-[1600px] px-8 py-10">
      <div ref={tabsRef} className="grid gap-4 lg:grid-cols-4">
        {workspacePages.map((item, index) => (
          <RevealItem key={item.key} hoverLift index={index} isVisible={areTabsVisible}>
            <Link
              href={item.href}
              className={`block rounded-[24px] border px-5 py-5 transition ${
                activePage === item.key
                  ? "border-fuchsia-300/30 bg-fuchsia-400/10"
                  : "border-white/10 bg-white/[0.03] hover:border-white/18"
              }`}
            >
              <p className="text-xs uppercase tracking-[0.24em] text-white/40">
                {item.label}
              </p>
              <p className="mt-3 text-sm leading-6 text-white/62">{item.description}</p>
            </Link>
          </RevealItem>
        ))}
      </div>
    </section>
  );
}

function ScoreFocusSection({
  selectedArchive,
  selectedInstrument,
  onSelectInstrument,
  selectedScore,
  availableInstruments,
  timeline,
  noteCount,
  scoreUrl,
  midiUrl,
  warnings,
}: {
  selectedArchive: ArchiveItem;
  selectedInstrument: InstrumentKey;
  onSelectInstrument: (instrument: InstrumentKey) => void;
  selectedScore: { title: string; scoreRows: ScoreRow[] };
  availableInstruments: InstrumentKey[];
  timeline: TimelineSegment[];
  noteCount: number;
  scoreUrl: string | null;
  midiUrl: string | null;
  warnings: string[];
}) {
  const { ref: scoreSectionRef, isVisible: isScoreSectionVisible } =
    useRevealOnce<HTMLDivElement>();

  return (
    <section className="mx-auto max-w-[1600px] px-8 py-14">
      <div ref={scoreSectionRef} className="grid gap-8 lg:grid-cols-[1.15fr_0.85fr]">
        <RevealItem hoverLift index={0} isVisible={isScoreSectionVisible}>
          <div className="overflow-hidden rounded-[30px] border border-white/10 bg-[#12151a]">
          <div className="border-b border-white/10 px-7 py-5">
            <p className="text-xs uppercase tracking-[0.28em] text-white/40">Notation</p>
            <h2 className="mt-3 text-3xl font-medium text-white">{selectedScore.title}</h2>
          </div>
          <div className="grid gap-0 lg:grid-cols-[1.05fr_0.95fr]">
            <div className="border-b border-white/10 p-7 lg:border-b-0 lg:border-r">
              <div className="rounded-[24px] border border-white/10 bg-[#0d0f12] p-6">
                <div className="flex items-center justify-between text-xs uppercase tracking-[0.24em] text-white/40">
                  <span>표기 미리보기</span>
                  <span>{selectedInstrument}</span>
                </div>
                <div className="mt-6 space-y-4">
                  {selectedScore.scoreRows.map((row) => (
                    <div
                      key={row.bar}
                      className="grid grid-cols-[48px_72px_1fr] gap-4 border-b border-white/6 pb-4 last:border-b-0"
                    >
                      <span className="text-sm text-white/36">{row.bar}</span>
                      <span className="font-medium text-fuchsia-100">{row.chord}</span>
                      <div>
                        <p className="text-sm leading-6 text-white/76">{row.note}</p>
                        <p className="mt-1 text-xs leading-5 text-white/42">{row.cue}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="space-y-4 p-7">
              <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-6">
                <p className="text-xs uppercase tracking-[0.24em] text-white/40">
                  구간 타임라인
                </p>
                <div className="mt-5 flex h-12 overflow-hidden rounded-full bg-white/6">
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
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-6">
                <p className="text-xs uppercase tracking-[0.24em] text-white/40">
                  악기 선택
                </p>
                <div className="mt-5 flex flex-wrap gap-2">
                  {availableInstruments.map((item) => (
                    <button
                      key={item}
                      type="button"
                      onClick={() => onSelectInstrument(item)}
                      className={`rounded-full px-4 py-2 text-sm transition ${
                        selectedInstrument === item
                          ? "border border-fuchsia-300/30 bg-fuchsia-400/12 text-fuchsia-100"
                          : "border border-white/10 bg-[#0d0f12] text-white/62 hover:text-white"
                      }`}
                    >
                      {item}
                    </button>
                  ))}
                </div>
              </div>
              <div className="rounded-[24px] border border-white/10 bg-white/[0.03] p-6">
                <p className="text-xs uppercase tracking-[0.24em] text-white/40">
                  결과 자산
                </p>
                <div className="mt-4 grid gap-3 sm:grid-cols-3">
                  <InfoPill label="노트 수" value={`${noteCount} events`} />
                  <InfoPill label="MusicXML" value={scoreUrl ? "준비됨" : "없음"} />
                  <InfoPill label="MIDI" value={midiUrl ? "준비됨" : "없음"} />
                </div>
                {(scoreUrl || midiUrl) ? (
                  <div className="mt-4 flex flex-wrap gap-3">
                    {scoreUrl ? (
                      <a
                        href={scoreUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:text-white"
                      >
                        MusicXML 열기
                      </a>
                    ) : null}
                    {midiUrl ? (
                      <a
                        href={midiUrl}
                        target="_blank"
                        rel="noreferrer"
                        className="rounded-full border border-white/12 px-4 py-2 text-sm text-white/70 transition hover:text-white"
                      >
                        MIDI 열기
                      </a>
                    ) : null}
                  </div>
                ) : null}
              </div>
              <div className="grid gap-3 sm:grid-cols-3">
                <InfoPill label="Track" value={selectedArchive.title} />
                <InfoPill label="Key" value={selectedArchive.key} />
                <InfoPill label="BPM" value={`${selectedArchive.bpm}`} />
              </div>
            </div>
          </div>
          </div>
        </RevealItem>

        <div className="space-y-4">
          <RevealItem hoverLift index={1} isVisible={isScoreSectionVisible}>
            <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
              <p className="text-sm font-medium text-white">페이지 역할</p>
              <p className="mt-4 text-sm leading-7 text-white/60">
                이 페이지는 연주 가능한 표기와 구간별 주석을 읽는 데 집중합니다. 악기별
                탐색보다 notation 자체를 오래 머무르며 보는 화면입니다.
              </p>
            </div>
          </RevealItem>
          <RevealItem hoverLift index={2} isVisible={isScoreSectionVisible}>
            <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
              <p className="text-sm font-medium text-white">현재 주목 지점</p>
              <div className="mt-4 space-y-3">
                {selectedArchive.highlights.map((highlight) => (
                  <div
                    key={highlight}
                    className="rounded-[20px] border border-white/10 bg-[#0d0f12] px-4 py-4 text-sm leading-6 text-white/68"
                  >
                    {highlight}
                  </div>
                ))}
              </div>
            </div>
          </RevealItem>
          {warnings.length > 0 ? (
            <RevealItem hoverLift index={3} isVisible={isScoreSectionVisible}>
              <div className="rounded-[28px] border border-amber-200/15 bg-amber-400/5 p-6">
                <p className="text-sm font-medium text-amber-100">분석 메모</p>
                <div className="mt-4 space-y-3">
                  {warnings.map((warning) => (
                    <div
                      key={warning}
                      className="rounded-[20px] border border-amber-100/10 bg-[#0d0f12] px-4 py-4 text-sm leading-6 text-white/68"
                    >
                      {warning}
                    </div>
                  ))}
                </div>
              </div>
            </RevealItem>
          ) : null}
        </div>
      </div>
    </section>
  );
}

function EffectsLabSection({
  selectedArchive,
  selectedInstrument,
  onSelectInstrument,
  availableInstruments,
  timeline,
  effectMessage,
}: {
  selectedArchive: ArchiveItem;
  selectedInstrument: InstrumentKey;
  onSelectInstrument: (instrument: InstrumentKey) => void;
  availableInstruments: InstrumentKey[];
  timeline: TimelineSegment[];
  effectMessage: string | null;
}) {
  const { ref: effectsSectionRef, isVisible: isEffectsSectionVisible } =
    useRevealOnce<HTMLDivElement>();

  return (
    <section className="mx-auto max-w-[1600px] px-8 py-14">
      <div ref={effectsSectionRef} className="grid gap-8 lg:grid-cols-[0.82fr_1.18fr]">
        <div className="space-y-4">
          <RevealItem hoverLift index={0} isVisible={isEffectsSectionVisible}>
            <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
              <p className="text-sm font-medium text-white">활성 파트</p>
              <div className="mt-5 flex flex-wrap gap-2">
                {availableInstruments.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => onSelectInstrument(item)}
                    className={`rounded-full px-4 py-2 text-sm transition ${
                      selectedInstrument === item
                        ? "border border-fuchsia-300/30 bg-fuchsia-400/12 text-fuchsia-100"
                        : "border border-white/10 bg-[#0d0f12] text-white/62 hover:text-white"
                    }`}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>
          </RevealItem>
          <RevealItem hoverLift index={1} isVisible={isEffectsSectionVisible}>
            <div className="rounded-[28px] border border-white/10 bg-white/[0.03] p-6">
              <p className="text-sm font-medium text-white">질감 키워드</p>
              <div className="mt-5 flex flex-wrap gap-2">
                {selectedArchive.moodTags.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-white/10 bg-[#0d0f12] px-3 py-1 text-xs uppercase tracking-[0.24em] text-white/56"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </RevealItem>
        </div>

        <RevealItem hoverLift index={2} isVisible={isEffectsSectionVisible}>
          <div className="rounded-[30px] border border-white/10 bg-[#12151a] p-7">
          <div className="flex items-end justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.28em] text-white/40">Profile</p>
              <h2 className="mt-3 text-3xl font-medium text-white">
                {selectedInstrument} 이펙터 프로파일
              </h2>
            </div>
            <span className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-white/62">
              {selectedArchive.title}
            </span>
          </div>

          <div className="mt-8 grid gap-5 lg:grid-cols-2">
            {selectedArchive.effectMetrics.map((item) => (
              <div
                key={item.label}
                className="rounded-[24px] border border-white/10 bg-white/[0.03] p-5"
              >
                <div className="flex items-center justify-between text-sm text-white/72">
                  <span>{item.label}</span>
                  <span className="text-fuchsia-100">{item.value.toFixed(2)}</span>
                </div>
                <div className="mt-3 h-2 overflow-hidden rounded-full bg-white/8">
                  <div
                    className="h-full rounded-full bg-gradient-to-r from-fuchsia-300 via-fuchsia-200 to-white"
                    style={{ width: `${item.value * 100}%` }}
                  />
                </div>
                <p className="mt-3 text-sm leading-6 text-white/54">{item.detail}</p>
              </div>
            ))}
          </div>

          <div className="mt-8 rounded-[24px] border border-white/10 bg-[#0d0f12] p-6">
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
            <p className="mt-5 text-sm leading-7 text-white/58">
              이 페이지는 파트 선택에 따라 질감 프로파일을 읽는 역할입니다. 악보보다
              청감적 해석을 우선하고, 비교분석 전에 사운드 성향을 정리합니다.
            </p>
            {effectMessage ? (
              <div className="mt-5 rounded-[20px] border border-fuchsia-300/15 bg-fuchsia-400/8 px-4 py-4 text-sm leading-6 text-fuchsia-100/82">
                {effectMessage}
              </div>
            ) : null}
          </div>
          </div>
        </RevealItem>
      </div>
    </section>
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
  const { ref: heroCardsRef, isVisible: areHeroCardsVisible } =
    useRevealOnce<HTMLDivElement>();

  return (
    <section id="top" className="relative overflow-hidden border-b border-white/10">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_top_left,rgba(217,70,239,0.14),transparent_30%),radial-gradient(circle_at_bottom_right,rgba(255,255,255,0.06),transparent_28%)]" />
      <div className="absolute inset-x-0 bottom-0 h-[340px] bg-[linear-gradient(to_top,rgba(255,255,255,0.02),transparent)]" />

      <div className="relative mx-auto grid min-h-[var(--lead-viewport-height)] max-w-[1600px] items-end gap-12 px-8 pb-16 pt-16 sm:pb-20 sm:pt-20 lg:grid-cols-[1.05fr_0.95fr]">
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

        <div ref={heroCardsRef} className="grid gap-5">
          <RevealItem hoverLift index={0} isVisible={areHeroCardsVisible}>
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
          </RevealItem>

          <RevealItem hoverLift index={1} isVisible={areHeroCardsVisible}>
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
          </RevealItem>
        </div>
      </div>
    </section>
  );
}

function ArchiveSection({
  hoveredArchiveId,
  selectedArchiveId,
  onHoverArchive,
  onInspectArchive,
  onPreviewArchive,
}: {
  hoveredArchiveId: string | null;
  selectedArchiveId: string;
  onHoverArchive: (archiveId: string | null) => void;
  onInspectArchive: (archiveId: string) => void;
  onPreviewArchive: (archiveId: string) => void;
}) {
  const { ref: archiveGridRef, isVisible: areArchiveCardsVisible } =
    useRevealOnce<HTMLDivElement>();

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

      <div ref={archiveGridRef} className="grid gap-6 lg:grid-cols-3">
        {archiveItems.map((item, index) => {
          const isSelected = item.id === selectedArchiveId;
          const isHovered = item.id === hoveredArchiveId;

          return (
            <RevealItem key={item.id} hoverLift index={index} isVisible={areArchiveCardsVisible}>
              <article>
              <button
                type="button"
                aria-label={`${item.title} 샘플 보기`}
                onClick={() => {
                  onInspectArchive(item.id);
                  onPreviewArchive(item.id);
                }}
                onMouseEnter={() => {
                  onHoverArchive(item.id);
                  onInspectArchive(item.id);
                }}
                onMouseLeave={() => onHoverArchive(null)}
                onFocus={() => {
                  onHoverArchive(item.id);
                  onInspectArchive(item.id);
                }}
                onBlur={() => onHoverArchive(null)}
                className={`group block w-full overflow-hidden rounded-[28px] border bg-white/5 text-left transition ${
                  isHovered
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
            </RevealItem>
          );
        })}
      </div>
    </section>
  );
}

function FeatureSection() {
  const { ref: featureGridRef, isVisible: areFeatureCardsVisible } =
    useRevealOnce<HTMLDivElement>();

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

        <div ref={featureGridRef} className="grid gap-6 lg:grid-cols-4">
          {features.map((item, index) => {
            const Icon = item.icon;

            return (
              <RevealItem
                key={item.title}
                hoverLift
                index={index}
                isVisible={areFeatureCardsVisible}
              >
                <article className="rounded-[26px] border border-white/10 bg-white/5 p-7">
                  <div className="flex h-12 w-12 items-center justify-center rounded-2xl bg-fuchsia-300/10 text-fuchsia-200">
                    <Icon className="h-5 w-5" />
                  </div>
                  <h3 className="mt-6 text-2xl font-medium text-white">{item.title}</h3>
                  <p className="mt-4 text-sm leading-7 text-white/62">
                    {item.description}
                  </p>
                </article>
              </RevealItem>
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
  availableInstruments,
  timeline,
  workspaceTracks,
  liveSummary,
}: {
  selectedArchive: ArchiveItem;
  selectedInstrument: InstrumentKey;
  onSelectInstrument: (instrument: InstrumentKey) => void;
  selectedScore: { title: string; scoreRows: ScoreRow[] };
  availableInstruments: InstrumentKey[];
  timeline: TimelineSegment[];
  workspaceTracks: WorkspaceTrack[];
  liveSummary: string;
}) {
  const { ref: analysisShowcaseRef, isVisible: isAnalysisShowcaseVisible } =
    useRevealOnce<HTMLDivElement>();

  return (
    <section id="analysis" className="mx-auto max-w-[1600px] px-8 py-24">
      <div ref={analysisShowcaseRef} className="grid gap-8 lg:grid-cols-[0.7fr_1.3fr]">
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

          <RevealItem hoverLift index={0} isVisible={isAnalysisShowcaseVisible}>
            <div className="rounded-[28px] border border-white/10 bg-white/5 p-7">
              <div className="flex items-center gap-3 text-sm text-white/46">
                <Sparkles className="h-4 w-4 text-fuchsia-200" />
                현재 선택된 파트
              </div>
              <div className="mt-6 flex flex-wrap gap-3">
                {availableInstruments.map((item) => (
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
          </RevealItem>

          <RevealItem hoverLift index={1} isVisible={isAnalysisShowcaseVisible}>
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
          </RevealItem>
        </div>

        <div className="space-y-8">
          <RevealItem hoverLift index={2} isVisible={isAnalysisShowcaseVisible}>
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
                    {workspaceTracks.some((track) => track.scoreUrl) ? "MusicXML Ready" : "Score Preview"}
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
                      <InfoPill
                        label="코드"
                        value={selectedScore.scoreRows[0]?.chord ?? "-"}
                      />
                      <InfoPill
                        label="강약"
                        value={selectedScore.scoreRows[0]?.cue ?? "구간 정보 준비 중"}
                      />
                      <InfoPill
                        label="주석"
                        value={liveSummary ? `${liveSummary} 결과 반영` : `${selectedInstrument} 중심 해석`}
                      />
                    </div>
                  </div>
                </div>
                <div className="border-t border-white/10 p-7">
                  <div className="flex items-center justify-between gap-4">
                    <div>
                      <p className="text-xs uppercase tracking-[0.24em] text-white/42">
                        Demucs Tracks
                      </p>
                      <p className="mt-2 text-sm text-white/56">
                        실제 분리된 stem과 원본 트랙을 여기서 바로 확인합니다.
                      </p>
                    </div>
                    <span className="rounded-full border border-white/10 bg-white/[0.03] px-4 py-2 text-sm text-white/62">
                      {workspaceTracks.length} tracks
                    </span>
                  </div>
                  <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                    {workspaceTracks.length > 0 ? (
                      workspaceTracks.map((track) => (
                        <div
                          key={track.id}
                          className="rounded-[22px] border border-white/10 bg-white/[0.03] p-5"
                        >
                          <div className="flex items-center justify-between gap-3">
                            <div>
                              <p className="text-sm font-medium text-white">{track.label}</p>
                              <p className="mt-1 text-xs uppercase tracking-[0.24em] text-white/42">
                                {track.source || "analysis"}
                              </p>
                            </div>
                            {track.instrumentKey ? (
                              <span className="rounded-full border border-fuchsia-300/20 bg-fuchsia-400/10 px-3 py-1 text-xs text-fuchsia-100">
                                {track.instrumentKey}
                              </span>
                            ) : null}
                          </div>
                          <p className="mt-4 text-sm leading-6 text-white/58">
                            {track.noteSummary}
                          </p>
                          {track.audioUrl ? (
                            <audio
                              controls
                              preload="none"
                              className="mt-4 w-full"
                              src={track.audioUrl}
                            />
                          ) : (
                            <div className="mt-4 rounded-[18px] border border-white/10 bg-[#0d0f12] px-4 py-3 text-sm text-white/44">
                              재생 가능한 오디오가 아직 없습니다.
                            </div>
                          )}
                        </div>
                      ))
                    ) : (
                      <div className="rounded-[22px] border border-white/10 bg-[#0d0f12] px-5 py-5 text-sm text-white/54">
                        분리된 트랙이 아직 준비되지 않았습니다.
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </RevealItem>
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
  originalAudioUrl,
  resynthAudioUrl,
  evaluation,
  compareSummary,
}: {
  comparePreview: ComparePreviewState;
  similarity: number;
  onToggleCard: (track: CompareTrack) => void;
  originalAudioUrl: string | null;
  resynthAudioUrl: string | null;
  evaluation: AnalysisResultResponse["evaluation"] | null;
  compareSummary: string;
}) {
  const { ref: compareGridRef, isVisible: areCompareCardsVisible } =
    useRevealOnce<HTMLDivElement>();

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

        <div ref={compareGridRef} className="grid gap-6 lg:grid-cols-[1fr_1fr_0.7fr]">
          <RevealItem hoverLift index={0} isVisible={areCompareCardsVisible}>
            <CompareCard
              title="원음원"
              subtitle="Original"
              isPlaying={comparePreview.open && comparePreview.activeTrack === "original" && comparePreview.isPlaying}
              onToggle={() => onToggleCard("original")}
              audioUrl={originalAudioUrl}
            />
          </RevealItem>
          <RevealItem hoverLift index={1} isVisible={areCompareCardsVisible}>
            <CompareCard
              title="재합성 음원"
              subtitle="Resynthesized"
              isPlaying={comparePreview.open && comparePreview.activeTrack === "resynth" && comparePreview.isPlaying}
              onToggle={() => onToggleCard("resynth")}
              audioUrl={resynthAudioUrl}
            />
          </RevealItem>
          <RevealItem hoverLift index={2} isVisible={areCompareCardsVisible}>
            <div className="rounded-[30px] border border-white/10 bg-white/5 p-7">
              <p className="text-xs uppercase tracking-[0.28em] text-white/42">Evaluation</p>
              <div className="mt-6 space-y-5">
                <MetricLine label="전체 유사도" value={similarity} />
                <MetricLine label="코드 정확도" value={clampMetric(evaluation?.chord_accuracy, 0.74)} />
                <MetricLine label="멜로디 일치도" value={clampMetric(evaluation?.melody_f1, 0.69)} />
                <MetricLine label="분리 품질" value={clampMetric(evaluation?.source_separation_quality, 0.77)} />
              </div>
              <div className="mt-8 rounded-[22px] border border-fuchsia-300/15 bg-fuchsia-400/8 px-5 py-4 text-sm leading-6 text-fuchsia-100/88">
                {compareSummary ||
                  "후렴 구간에서의 공간계 표현과 보컬 레이어는 원곡과 매우 유사하게 재현되었지만, 저역 베이스의 어택은 다소 부드럽게 추정되었습니다."}
              </div>
            </div>
          </RevealItem>
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
  audioUrl,
}: {
  title: string;
  subtitle: string;
  isPlaying: boolean;
  onToggle: () => void;
  audioUrl: string | null;
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
      {audioUrl ? (
        <audio controls preload="none" className="mt-5 w-full" src={audioUrl} />
      ) : (
        <div className="mt-5 rounded-[18px] border border-white/10 bg-white/[0.03] px-4 py-3 text-sm text-white/42">
          비교용 오디오가 아직 준비되지 않았습니다.
        </div>
      )}
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
  onStart,
}: {
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
          <Link href="/archive" className="transition hover:text-white">
            샘플 아카이브
          </Link>
          <button
            type="button"
            onClick={onStart}
            className="transition hover:text-white"
          >
            분석 시작
          </button>
          <Link href="/compare" className="transition hover:text-white">
            비교 보기
          </Link>
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
  jobStatus,
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
  jobStatus: JobStatusResponse | null;
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
  const renderedStages =
    draft.sourceMode !== "sample" && jobStatus?.stages.length
      ? jobStatus.stages.map((stage) => ({
          title: stage.label,
          detail: stage.message || stage.description || stage.error || "대기 중",
          status:
            stage.status === "completed"
              ? "done"
              : stage.status === "running"
                ? "current"
                : stage.status === "failed"
                  ? "failed"
                  : stage.status === "skipped"
                    ? "skipped"
                    : "pending",
        }))
      : requestStages.map((stage, index) => ({
          title: stage.title,
          detail: stage.detail,
          status:
            index < requestStageIndex
              ? "done"
              : index === requestStageIndex
                ? "current"
                : "pending",
        }));

  return (
    <DialogShell
      title="분석 요청 창"
      description="06ver에서는 샘플 모드는 기존 워크스페이스 흐름을 유지하고, 파일 업로드와 링크 입력은 02ver 백엔드의 실제 job/result API로 연결됩니다."
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
                    선택한 파일은 실제로 `02ver/backend` 분석 파이프라인으로 전송됩니다.
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
                다운로드 가능한 오디오 링크라면 실제 백엔드에서 받아서 분석 작업을 시작합니다.
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
                {draft.sourceMode === "sample"
                  ? "선택한 샘플과 옵션을 기준으로 워크스페이스 기본값을 맞춥니다."
                  : "선택한 소스와 옵션을 기준으로 실제 백엔드 job을 생성하고, 분석 상태를 이 창에서 바로 확인합니다."}
              </div>
            ) : null}

            {requestStatus === "preparing" ? (
              <div className="mt-5 space-y-4">
                {renderedStages.map((stage) => {
                  return (
                    <div
                      key={stage.title}
                      className={`rounded-[22px] border px-4 py-4 ${
                        stage.status === "done"
                          ? "border-fuchsia-300/20 bg-fuchsia-400/8"
                          : stage.status === "current"
                            ? "border-white/14 bg-[#0d0f12]"
                            : stage.status === "failed"
                              ? "border-rose-300/20 bg-rose-400/8"
                            : "border-white/8 bg-transparent"
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`flex h-9 w-9 items-center justify-center rounded-full ${
                            stage.status === "done"
                              ? "bg-fuchsia-300/20 text-fuchsia-100"
                              : stage.status === "current"
                                ? "bg-white/[0.06] text-white"
                                : stage.status === "failed"
                                  ? "bg-rose-400/12 text-rose-100"
                                : "bg-white/[0.04] text-white/34"
                          }`}
                        >
                          {stage.status === "done" ? (
                            <Check className="h-4 w-4" />
                          ) : stage.status === "failed" ? (
                            <X className="h-4 w-4" />
                          ) : (
                            <Disc3
                              className={`h-4 w-4 ${
                                stage.status === "current" ? "animate-spin" : ""
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
                    {draft.sourceMode === "sample"
                      ? "선택한 샘플 기준으로 워크스페이스가 준비되었습니다."
                      : "실제 분석 결과가 준비되어 각 페이지와 연결되었습니다."}
                  </p>
                  <p className="mt-2 text-sm leading-6 text-fuchsia-100/76">
                    {draft.sourceMode === "sample"
                      ? "이제 작업 창으로 이동해 샘플 기준 워크스페이스를 이어서 확인할 수 있습니다."
                      : "악기별 탐색, 악보, 이펙터, 비교분석 페이지에서 방금 완료된 결과를 바로 확인할 수 있습니다."}
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
    <div className="rounded-[20px] border border-white/10 bg-white/[0.03] px-4 py-3.5">
      <p className="text-[11px] uppercase tracking-[0.24em] text-white/38">{label}</p>
      <p className="mt-1.5 text-base font-medium text-white/74">{value}</p>
    </div>
  );
}

function ComparePreviewDialog({
  activeTrack,
  activeSection,
  isPlaying,
  progress,
  selectedArchive,
  timeline,
  originalAudioUrl,
  resynthAudioUrl,
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
  timeline: TimelineSegment[];
  originalAudioUrl: string | null;
  resynthAudioUrl: string | null;
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
      description="원음원과 재합성 오디오를 같은 문맥에서 비교하도록 만든 미리듣기 창입니다. 06ver에서는 실제 분석 결과의 오디오 URL이 있으면 여기에도 바로 연결됩니다."
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
            {(activeTrack === "original" ? originalAudioUrl : resynthAudioUrl) ? (
              <audio
                controls
                preload="none"
                className="mt-5 w-full"
                src={activeTrack === "original" ? originalAudioUrl ?? undefined : resynthAudioUrl ?? undefined}
              />
            ) : null}
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
