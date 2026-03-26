"use client";

export type JobState = "queued" | "processing" | "completed" | "failed";
export type StageState = "pending" | "running" | "completed" | "failed" | "skipped";

export type MetaInfo = {
  title: string;
  duration: number;
  bpm: number | null;
  key: string | null;
  sample_rate: number | null;
  analysis_summary: string | null;
  waveform: number[];
  spectrogram: number[][];
  mel_spectrogram: number[][];
};

export type SectionItem = {
  start: number;
  end: number;
  label: string;
  chords: string[];
  dynamics: string | null;
  instruments: string[];
  effects: string[];
  source: string | null;
  confidence: number | null;
};

export type TrackItem = {
  instrument: string;
  audio_url: string | null;
  score_url: string | null;
  midi_url: string | null;
  note_summary: string | null;
  source: string | null;
};

export type NoteEvent = {
  start: number;
  end: number;
  pitch: number;
  velocity: number | null;
};

export type GuitarEffectsInfo = {
  available: boolean;
  drive: number | null;
  space: number | null;
  modulation: number | null;
  tone_brightness: number | null;
  sustain: number | null;
  transient_focus: number | null;
  labels: string[];
  message: string | null;
};

export type EvaluationInfo = {
  similarity: number | null;
  chord_accuracy: number | null;
  melody_f1: number | null;
  source_separation_quality: number | null;
  effect_confidence: number | null;
};

export type CompareInfo = {
  original_audio_url: string | null;
  resynth_audio_url: string | null;
  waveform_diff: number[];
  spectrum_diff: number[];
};

export type AnalysisResultResponse = {
  job_id: string;
  status: JobState;
  meta: MetaInfo;
  sections: SectionItem[];
  tracks: TrackItem[];
  melody_notes: NoteEvent[];
  guitar_effects: GuitarEffectsInfo;
  evaluation: EvaluationInfo;
  compare: CompareInfo;
  warnings: string[];
  errors: string[];
};

export type PipelineStageStatus = {
  id: string;
  label: string;
  description: string;
  status: StageState;
  message: string;
  error: string;
};

export type JobStatusResponse = {
  job_id: string;
  status: JobState;
  source_type: string;
  title: string;
  created_at: string;
  updated_at: string;
  stages: PipelineStageStatus[];
  warnings: string[];
  error_message: string | null;
  result_available: boolean;
};

type CreateAnalysisJobParams = {
  file?: File | null;
  sourceLink?: string;
  title?: string;
};

function normalizeApiBaseUrl() {
  const envValue = process.env.NEXT_PUBLIC_MUSIC_ARCHIVE_API_BASE_URL?.trim();
  return (envValue || "http://localhost:8000/api").replace(/\/+$/, "");
}

function getBackendOrigin(apiBaseUrl: string) {
  return apiBaseUrl.endsWith("/api")
    ? apiBaseUrl.slice(0, -4)
    : apiBaseUrl;
}

async function getErrorMessage(response: Response) {
  const text = await response.text();

  if (!text) {
    return `요청에 실패했습니다. (${response.status})`;
  }

  try {
    const payload = JSON.parse(text) as {
      detail?: string;
      message?: string;
      error_message?: string;
    };

    return (
      payload.detail ||
      payload.message ||
      payload.error_message ||
      text ||
      `요청에 실패했습니다. (${response.status})`
    );
  } catch {
    return text;
  }
}

async function fetchJson<T>(url: string, init?: RequestInit) {
  let response: Response;

  try {
    response = await fetch(url, {
      ...init,
      cache: "no-store",
    });
  } catch {
    throw new Error(
      "백엔드에 연결할 수 없습니다. `02ver/backend`가 localhost:8000에서 실행 중인지 확인해 주세요.",
    );
  }

  if (!response.ok) {
    throw new Error(await getErrorMessage(response));
  }

  return (await response.json()) as T;
}

export function getAnalysisApiBaseUrl() {
  return normalizeApiBaseUrl();
}

export function resolveAnalysisMediaUrl(path: string | null | undefined) {
  if (!path) {
    return null;
  }

  if (/^https?:\/\//i.test(path)) {
    return path;
  }

  const origin = getBackendOrigin(normalizeApiBaseUrl());
  return `${origin}${path.startsWith("/") ? path : `/${path}`}`;
}

export async function createAnalysisJob({
  file,
  sourceLink,
  title,
}: CreateAnalysisJobParams) {
  const formData = new FormData();

  if (file) {
    formData.set("file", file);
  }

  if (sourceLink?.trim()) {
    formData.set("source_link", sourceLink.trim());
  }

  if (title?.trim()) {
    formData.set("title", title.trim());
  }

  return fetchJson<JobStatusResponse>(`${normalizeApiBaseUrl()}/jobs`, {
    method: "POST",
    body: formData,
  });
}

export async function getAnalysisJobStatus(jobId: string) {
  return fetchJson<JobStatusResponse>(`${normalizeApiBaseUrl()}/jobs/${jobId}`);
}

export async function getAnalysisResult(jobId: string) {
  return fetchJson<AnalysisResultResponse>(`${normalizeApiBaseUrl()}/results/${jobId}`);
}
