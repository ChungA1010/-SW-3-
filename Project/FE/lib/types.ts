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
  source_id?: number;
  source_name: string;
  predicted_effect: string;
  predicted_effect_display_name: string;
  confidence: string;
  guitar_stem_url: string;
  original_audio_url: string;
};

// ----------------------------------------------------
// 🎸 피드백 관련 타입 정의
// ----------------------------------------------------

export interface FeatureComparison {
  feature: string;
  reference_value: number;
  copy_value: number;
  delta: number;
  percent_change: number;
  expected_direction: string;
  meaning: string;
}

export interface AxisFeedback {
  axis: string;
  reference_amount: number;
  copy_amount: number;
  difference: number;
  similarity: number;
  action: string;
  message: string;
  features: FeatureComparison[];
}

export interface EffectFeedback {
  overall_similarity: number;
  axes: AxisFeedback[];
}

//Tone과 Timeseries 뎁스가 사라지고 필드가 바로 노출되도록 수정
export interface PlayingFeedback {
  pitch_score: number;
  rhythm_score: number;
  combined_score: number;
  grade: string;
  pitch_reliable: boolean; // 추가됨
  pitch_mae_semitone: number;
  onset_mae_ms: number;
  issues: string[];
  suggestions: string[];
  strengths: string[];
}

export interface UnifiedFeedback {
  reference_path: string;
  copy_path: string;
  effect_feedback: EffectFeedback;
  playing_feedback: PlayingFeedback;
}

export interface FeedbackResponse {
  success: boolean;
  message: string;
  feedback?: UnifiedFeedback;
}

// ================  로그 관련 응답 처리 =====================
export interface PredictedEffect {
  dist: string;
  delay: string;
  phase: string;
}

export interface AnalysisLog {
  track_id: number;
  source_info: {
    source_id: string;
    name: string;
  };
  file_path: string | null;
  predicted_effect: PredictedEffect | null;
}

export interface LogsResponse {
  success: boolean;
  logs: AnalysisLog[];
}