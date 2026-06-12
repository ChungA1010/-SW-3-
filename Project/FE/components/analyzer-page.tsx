"use client";

import Link from "next/link";
import { useMemo, useState, useEffect } from "react";

import { analyzeFile, analyzeYoutube } from "@/lib/api";
import { Header } from "./Header";
import type { AnalysisResponse, TimelineSegment } from "@/lib/types";

import { ResultView } from "./ResultView";

type Mode = "file" | "youtube";

const LOADING_STEPS = [
  "실행 중....",
];



export function AnalyzerPage() {
  //추가된 상태(시작 시간, 종료 시간, 이름)
  const [startSec, setStartSec] = useState<string>("");
  const [endSec, setEndSec] = useState<string>("");
  const [customName, setCustomName] = useState<string>("");

  const [mode, setMode] = useState<Mode>("file");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [youtubeUrl, setYoutubeUrl] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [errorMessage, setErrorMessage] = useState("");

  useEffect(() => {
    const savedResult = localStorage.getItem("analysis_result");
    if (savedResult) {
      try {
        setResult(JSON.parse(savedResult));
      } catch (e) {
        console.error("데이터 복구 실패:", e);
      }
    }
  }, []);

  useEffect(() => {
    if (result) {
      localStorage.setItem("analysis_result", JSON.stringify(result));
    }
  }, [result]);



  async function handleAnalyze() {
    localStorage.removeItem("analysis_result");
    setErrorMessage("");
    setResult(null);
    setIsLoading(true);

    const parsedStart = startSec !== "" ? Number(startSec) : undefined;
    const parsedEnd = endSec !== "" ? Number(endSec) : undefined;

    try {
      const response =
        mode === "file"
          ? await analyzeFile(
            assertFile(selectedFile),
            parsedStart,
            parsedEnd,
            customName.trim()
          )
          : await analyzeYoutube(
            youtubeUrl.trim(),
            parsedStart,
            parsedEnd,
            customName.trim());
      setResult(response);
    } catch (error) {
      setErrorMessage(error instanceof Error ? error.message : "오류가 발생했습니다.");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <>
      <Header onReset={() => setResult(null)} />
      <main className="archive-shell">

        <section id="input" className="workspace-grid workspace-grid-lowered">
          <article className="workspace-card workspace-card-strong">

            {/* 모드 전환 탭 버튼 영역 */}
            {/* 📌 수정된 탭 버튼 영역 */}
            <div className="mode-pills" style={{ marginBottom: '2rem' }}>
              <button
                type="button"
                className={`mode-pill ${mode === "file" ? "active" : ""}`}
                onClick={() => setMode("file")}
              >
                오디오 파일
              </button>
              <button
                type="button"
                className={`mode-pill ${mode === "youtube" ? "active" : ""}`}
                onClick={() => setMode("youtube")}
              >
                유튜브 링크
              </button>
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

                <label className="field-label">
                  <span>파일 이름 (선택)</span>
                  <input
                    className="field-input"
                    type="text"
                    value={customName}
                    onChange={(e) => setCustomName(e.target.value)}
                    placeholder="미입력 시 임의의 파일명 사용"
                  />
                </label>

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
                  <span>파일 이름 (선택)</span>
                  <input
                    className="field-input"
                    type="text"
                    value={customName}
                    onChange={(e) => setCustomName(e.target.value)}
                    placeholder="미입력 시 임의의 파일명 사용"
                  />
                </label>
              </div>
            )}

            <div className="field-stack" style={{ display: 'flex', gap: '1rem', marginTop: '1rem' }}>
              <label className="field-label" style={{ flex: 1 }}>
                <span>시작 시간 (초)</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.1"
                  min="0"
                  value={startSec}
                  onChange={(e) => setStartSec(e.target.value)}
                  placeholder="0.0"
                />
              </label>
              <label className="field-label" style={{ flex: 1 }}>
                <span>종료 시간 (초)</span>
                <input
                  className="field-input"
                  type="number"
                  step="0.1"
                  min="0"
                  value={endSec}
                  onChange={(e) => setEndSec(e.target.value)}
                  placeholder="10.0"
                />
              </label>
            </div>

            <button
              className="launch-button"
              type="button"
              disabled={isLoading || (mode === "file" ? !selectedFile : !youtubeUrl.trim())}
              onClick={handleAnalyze}
              style={{ marginTop: '1.5rem' }}
            >
              {isLoading ? "분석 중..." : "분석 시작"}
            </button>

            {errorMessage ? <div className="archive-error-box">{errorMessage}</div> : null}
          </article>

          {/* ... 상태 및 결과 섹션 유지 ... */}
          <article className="workspace-card">
            {isLoading ? (
              <div style={{ textAlign: 'center', padding: '5rem 0' }}>
                <h3 style={{ marginBottom: '1rem' }}>분석을 진행하고 있습니다...</h3>
              </div>
            ) : result ? (
              <ResultView result={result} />
            ) : (
              <EmptyState />
            )}
          </article>

          {/* ... 상태 및 결과 섹션 유지 ... */}
        </section>
      </main >
    </>
  );
}



function assertFile(file: File | null): File {
  if (!file) {
    throw new Error("파일을 먼저 선택하세요.");
  }
  return file;
}


function EmptyState() {
  return (
    <div className="empty-result-card">
      <h4>결과가 여기에 표시됩니다.</h4>
    </div>
  );
}
