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
  predicted_family: string;
  predicted_family_display_name: string;
  fine_scores: ScoreItem[];
  family_scores: ScoreItem[];
  timeline_segments: TimelineSegment[];
  audio_duration_sec: number;
  analysis_window_sec: number;
  analysis_hop_sec: number;
  guitar_stem_url: string;
  original_audio_url: string;
  processing_time_sec: number;
  demucs_stem_used: string;
  model_name: string;
};
