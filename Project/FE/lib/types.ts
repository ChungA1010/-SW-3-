export type ScoreItem = {
  label: string;
  display_name: string;
  score: number;
};

export type TimelineSegment = {
  start_sec: number;
  end_sec: number;
  duration_sec: number;
  effect_label: string;
  effect_display_name: string;
  family_label: string;
  family_display_name: string;
  confidence: number;
};

export type AnalysisResponse = {
  success: boolean;
  source_type: "file" | "youtube";
  source_name: string;
  predicted_effect: string;
  predicted_effect_display_name: string;
  confidence: string;         // 백엔드에서 "8.55%" 처럼 문자열로 넘어오므로 string으로 변경
  guitar_stem_url: string;
  original_audio_url: string;
};

export type SimpleResponse = {
  success: boolean;
  message?: string; // 백엔드에서 보내주는 안내 메시지 (선택 사항)
};