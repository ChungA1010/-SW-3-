import type { AnalysisResponse, SimpleResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";


async function handleResponse(response: Response): Promise<AnalysisResponse> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? "요청 처리 중 오류가 발생했습니다.");
  }

  return (await response.json()) as AnalysisResponse;
}

export async function analyzeFile(
  file: File,
  startSec?: number,
  endSec?: number,
  customName?: string
): Promise<AnalysisResponse> {
  const formData = new FormData();

  formData.append("file", file);
  formData.append("name", customName || file.name);

  if (startSec !== undefined) formData.append("start_sec", startSec.toString());
  if (endSec !== undefined) formData.append("end_sec", endSec.toString());

  const response = await fetch(`${API_BASE_URL}/ai/predict/`, {
    method: "POST",
    body: formData,
  });

  const data = await handleResponse(response);
  return withAbsoluteUrls(data);
}


// src/lib/api.ts

export async function analyzeYoutube(
  youtubeUrl: string,
  startSec?: number,
  endSec?: number
): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/ai/predict/`, { // 동일한 AI 파이프라인으로 쏨!
    method: "POST",
    headers: {
      "Content-Type": "application/json", // JSON 형식 명시
    },
    body: JSON.stringify({
      url: youtubeUrl,  // 백엔드 form.cleaned_data['url']과 매핑
      start_sec: startSec,
      end_sec: endSec
    }),
  });

  const data = await handleResponse(response);
  return withAbsoluteUrls(data);
}


function withAbsoluteUrls(data: AnalysisResponse): AnalysisResponse {

  if (!data.success) return data;

  return {
    ...data,
    guitar_stem_url: `${API_BASE_URL}${data.guitar_stem_url}`,
    original_audio_url: `${API_BASE_URL}${data.original_audio_url}`,
  };
}


export async function getFeedback(file: File, sourceID: number): Promise<SimpleResponse> {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("source_id", sourceID.toString());

  const response = await fetch(`${API_BASE_URL}/ai/feedback/`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string; message?: string } | null;
    throw new Error(payload?.detail ?? payload?.message ?? "요청 처리 중 오류가 발생했습니다.");
  }

  return (await response.json()) as SimpleResponse;
}
