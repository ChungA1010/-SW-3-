"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { analyzeFile, analyzeYoutube } from "@/lib/api";
import type { AnalysisResponse, TimelineSegment } from "@/lib/types";

type Mode = "file" | "youtube";

const LOADING_STEPS = [
  "오디오 준비",
  "Demucs 분리",
  "기타 stem 선택",
  "GRU 추론",
  "결과 정리",
];

export function AnalyzerPage() {
  const [mode, setMode] = useState<Mode>("file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  const activeStepIndex = useMemo(() => {
    if (!isLoading) {
      return -1;
    }
    return Math.floor(Date.now() / 900) % LOADING_STEPS.length;
  }, [isLoading]);

  async function handleAnalyze() {
    setErrorMessage("");
    setResult(null);
    setIsLoading(true);

    try {
      const response =
        mode === "file"
          ? await analyzeFile(assertFile(selectedFile))
          : await analyzeYoutube(youtubeUrl.trim());
      setResult(response);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main className="archive-shell">
      <Header />

      <section className="archive-hero archive-hero-compact">
        <div className="archive-hero-copy">
          <p className="section-eyebrow">입력</p>
          <h2>
            파일이나 링크로
            <br />
            바로 분석합니다.
          </h2>
        </div>

        <div className="hero-preview-card">
          <div className="hero-preview-panel">
            <p className="section-eyebrow">현재 구성</p>
            <h3>FastAPI + Demucs + GRU</h3>
            <div className="hero-preview-list">
              <PreviewRow label="입력" value="파일 / 유튜브" />
              <PreviewRow label="분리" value="Demucs" />
              <PreviewRow label="추론" value="GRU" />
              <PreviewRow label="출력" value="점수 / 구간 / 오디오" />
            </div>
          </div>
        </div>
      </section>

      <section id="input" className="workspace-grid workspace-grid-lowered">
        <article className="workspace-card workspace-card-strong">
          <div className="workspace-card-head">
            <div>
              <p className="section-eyebrow">사용자 입력</p>
              <h3>분석 요청</h3>
            </div>
            <div className="mode-pills">
              <button
                className={`mode-pill ${mode === "file" ? "active" : ""}`}
                onClick={() => setMode("file")}
                type="button"
              >
                파일
              </button>
              <button
                className={`mode-pill ${mode === "youtube" ? "active" : ""}`}
                onClick={() => setMode("youtube")}
                type="button"
              >
                유튜브
              </button>
            </div>
          </div>

          {mode === "file" ? (
            <div className="field-stack">
              <label className="field-label">
                <span>오디오 파일</span>
                <input
                  key="file-input"
                  className="field-input"
                  type="file"
                  accept=".wav,.mp3,.flac,.m4a,.ogg,.aac"
                  onChange={(event) => setSelectedFile(event.target.files?.[0] ?? null)}
                />
              </label>
              <div className="source-detail-card">
                <p className="source-detail-title">선택 파일</p>
                <p className="source-detail-value">
                  {selectedFile ? selectedFile.name : "선택된 파일이 없습니다."}
                </p>
                <p className="source-detail-subtle">
                  {selectedFile
                    ? `${(selectedFile.size / 1024 / 1024).toFixed(2)} MB`
                    : "wav, mp3, flac, m4a, ogg, aac"}
                </p>
              </div>
            </div>
          ) : (
            <div className="field-stack">
              <label className="field-label">
                <span>유튜브 링크</span>
                <input
                  key="youtube-input"
                  className="field-input"
                  type="url"
                  value={youtubeUrl}
                  onChange={(event) => setYoutubeUrl(event.target.value)}
                  placeholder="https://www.youtube.com/watch?v=..."
                />
              </label>
              <div className="source-detail-card">
                <p className="source-detail-title">입력 링크</p>
                <p className="source-detail-value">{youtubeUrl.trim() || "링크를 입력하세요."}</p>
              </div>
            </div>
          )}

          <button
            className="launch-button"
            type="button"
            disabled={isLoading || (mode === "file" ? !selectedFile : !youtubeUrl.trim())}
            onClick={handleAnalyze}
          >
            {isLoading ? "분석 중..." : "분석 시작"}
          </button>

          {errorMessage ? <div className="archive-error-box">{errorMessage}</div> : null}
        </article>

        <article className="workspace-card">
          <div className="workspace-card-head">
            <div>
              <p className="section-eyebrow">진행 상태</p>
              <h3>현재 단계</h3>
            </div>
            <span className="status-chip">{isLoading ? "진행 중" : "대기"}</span>
          </div>

          <div className="status-list">
            {LOADING_STEPS.map((step, index) => (
              <div
                className={`status-item ${isLoading && activeStepIndex === index ? "active" : ""} ${!isLoading && result ? "done" : ""}`}
                key={step}
              >
                <span className="status-index">{index + 1}</span>
                <div>
                  <p className="status-title">{step}</p>
                </div>
              </div>
            ))}
          </div>
        </article>
      </section>

      <section id="result" className="result-section">
        <div className="section-heading">
          <p className="section-eyebrow">결과</p>
          <h3>분석 결과</h3>
        </div>

        {result ? <ResultView result={result} /> : <EmptyState />}
      </section>
    </main>
  );
}

function Header() {
  return (
    <header className="archive-header">
      <div className="archive-brand">
        <span className="archive-brand-mark" />
        <div>
          <p className="archive-brand-eyebrow">기타 이펙터</p>
          <h1>이펙터 분석기</h1>
        </div>
      </div>
      <nav className="archive-nav">
        <Link href="/">입력</Link>
        <Link href="/overview">개요</Link>
      </nav>
    </header>
  );
}

function assertFile(file: File | null): File {
  if (!file) {
    throw new Error("파일을 먼저 선택하세요.");
  }
  return file;
}

function PreviewRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="preview-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="empty-result-card">
      <p className="section-eyebrow">대기</p>
      <h4>결과가 여기에 표시됩니다.</h4>
    </div>
  );
}

function ResultView({ result }: { result: AnalysisResponse }) {
  const topEffects = result.fine_scores.slice(0, 4);
  const topFamilies = result.family_scores.slice(0, 4);

  return (
    <div className="result-layout">
      <div className="result-overview-grid">
        <article className="result-main-card">
          <div className="result-main-top">
            <div>
              <p className="section-eyebrow">대표 결과</p>
              <h4>{result.predicted_effect_display_name}</h4>
              <p className="result-copy">원본: {result.source_name}</p>
            </div>
            <div className="result-meta-pills">
              <span>{result.source_type === "file" ? "파일" : "유튜브"}</span>
              <span>{result.predicted_family_display_name}</span>
              <span>{result.processing_time_sec.toFixed(2)}초</span>
            </div>
          </div>

          <div className="result-summary-grid">
            <SummaryCell label="모델" value={result.model_name.toUpperCase()} />
            <SummaryCell label="stem" value={result.demucs_stem_used} />
            <SummaryCell label="길이" value={`${result.audio_duration_sec.toFixed(1)}초`} />
          </div>

          <div className="result-audio-card">
            <p className="section-eyebrow">분리된 기타</p>
            <audio controls src={result.guitar_stem_url}>
              브라우저가 오디오 재생을 지원하지 않습니다.
            </audio>
          </div>
        </article>

        <article className="result-side-card result-side-card-compact">
          <ScoreBlock title="이펙터 점수" items={topEffects} />
        </article>

        <article className="result-side-card result-side-card-compact">
          <ScoreBlock title="계열 점수" items={topFamilies} />
        </article>
      </div>

      <article className="result-main-card timeline-card">
        <div className="timeline-top">
          <div>
            <p className="section-eyebrow">시계열 분석</p>
            <h3>구간별 이펙터</h3>
          </div>
          <div className="timeline-meta">
            <TimelineMetaCard label="전체 길이" value={`${result.audio_duration_sec.toFixed(1)}초`} />
            <TimelineMetaCard label="분석 윈도우" value={`${result.analysis_window_sec.toFixed(1)}초`} />
            <TimelineMetaCard label="이동 간격" value={`${result.analysis_hop_sec.toFixed(1)}초`} />
            <TimelineMetaCard label="구간 수" value={`${result.timeline_segments.length}개`} />
          </div>
        </div>

        <TimelineBar
          segments={result.timeline_segments}
          totalDuration={result.audio_duration_sec}
        />

        <div className="timeline-summary-row">
          <span>대표 구간 {result.timeline_segments.length}개</span>
          <span>가장 강한 이펙터: {result.predicted_effect_display_name}</span>
        </div>

        <div className="timeline-table-card">
          <div className="timeline-header-row">
            <span>구간</span>
            <span>이펙터</span>
            <span>신뢰도</span>
          </div>

          <div className="timeline-list timeline-list-scroll">
            {result.timeline_segments.map((segment, index) => (
              <TimelineRow key={`${segment.start_sec}-${segment.end_sec}-${index}`} segment={segment} />
            ))}
          </div>
        </div>
      </article>
    </div>
  );
}

function SummaryCell({ label, value }: { label: string; value: string }) {
  return (
    <div className="summary-cell">
      <p>{label}</p>
      <strong>{value}</strong>
    </div>
  );
}

function ScoreBlock({ title, items }: { title: string; items: AnalysisResponse["fine_scores"] }) {
  return (
    <section className="score-panel">
      <div className="workspace-card-head">
        <div>
          <p className="section-eyebrow">점수</p>
          <h3>{title}</h3>
        </div>
      </div>
      <div className="score-stack">
        {items.map((item) => (
          <div className="score-item" key={item.label}>
            <div className="score-item-top">
              <span>{item.display_name}</span>
              <strong>{(item.score * 100).toFixed(1)}%</strong>
            </div>
            <div className="score-track">
              <div className="score-fill" style={{ width: `${Math.max(item.score * 100, 1)}%` }} />
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}

function TimelineBar({
  segments,
  totalDuration,
}: {
  segments: TimelineSegment[];
  totalDuration: number;
}) {
  return (
    <div className="timeline-bar-wrap">
      <div className="timeline-ruler">
        <span>00:00</span>
        <span>{formatTime(totalDuration)}</span>
      </div>
      <div className="timeline-bar">
        {segments.map((segment, index) => {
          const width = totalDuration > 0 ? (segment.duration_sec / totalDuration) * 100 : 0;
          return (
            <div
              className={`timeline-bar-segment timeline-color-${index % 6}`}
              key={`${segment.start_sec}-${segment.end_sec}-${index}`}
              style={{ width: `${Math.max(width, 1.5)}%` }}
              title={`${formatTime(segment.start_sec)} - ${formatTime(segment.end_sec)}  ${segment.effect_display_name}`}
            />
          );
        })}
      </div>
      <div className="timeline-legend">
        {segments.slice(0, 6).map((segment, index) => (
          <div className="timeline-legend-item" key={`${segment.effect_label}-${index}`}>
            <span className={`timeline-legend-dot timeline-color-${index % 6}`} />
            <span>{segment.effect_display_name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function TimelineRow({ segment }: { segment: TimelineSegment }) {
  return (
    <div className="timeline-row">
      <div className="timeline-time">
        <strong>{formatTime(segment.start_sec)} - {formatTime(segment.end_sec)}</strong>
        <span>{segment.duration_sec.toFixed(1)}초</span>
      </div>
      <div className="timeline-effect">
        <strong>{segment.effect_display_name}</strong>
        <span>{segment.family_display_name}</span>
      </div>
      <div className="timeline-confidence">
        <span>신뢰도</span>
        <strong>{(segment.confidence * 100).toFixed(1)}%</strong>
      </div>
    </div>
  );
}

function formatTime(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(total / 60);
  const remain = total % 60;
  return `${String(minutes).padStart(2, "0")}:${String(remain).padStart(2, "0")}`;
}

function TimelineMetaCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="timeline-meta-card">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
