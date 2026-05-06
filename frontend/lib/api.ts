import type { AnalysisResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function handleResponse(response: Response): Promise<AnalysisResponse> {
  if (!response.ok) {
    const payload = (await response.json().catch(() => null)) as { detail?: string } | null;
    throw new Error(payload?.detail ?? "요청 처리 중 오류가 발생했습니다.");
  }

  return (await response.json()) as AnalysisResponse;
}

export async function analyzeFile(file: File): Promise<AnalysisResponse> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(`${API_BASE_URL}/api/analyze/file`, {
    method: "POST",
    body: formData,
  });

  const data = await handleResponse(response);
  return withAbsoluteUrls(data);
}

export async function analyzeYoutube(youtubeUrl: string): Promise<AnalysisResponse> {
  const response = await fetch(`${API_BASE_URL}/api/analyze/youtube`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ youtube_url: youtubeUrl }),
  });

  const data = await handleResponse(response);
  return withAbsoluteUrls(data);
}

function withAbsoluteUrls(data: AnalysisResponse): AnalysisResponse {
  return {
    ...data,
    guitar_stem_url: `${API_BASE_URL}${data.guitar_stem_url}`,
    original_audio_url: `${API_BASE_URL}${data.original_audio_url}`,
  };
}

