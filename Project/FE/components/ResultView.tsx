"use client";

import { useState } from "react";
import type { AnalysisResponse } from "@/lib/types"; // 타입 임포트
import { FeedbackRecorder } from "./FeedbackRecorder"; // 방금 만든 파일 임포트

// 📌 헬퍼 함수: 이펙터 문자열 값("off", "50", "100")을 width 퍼센트로 변환
const getBarWidth = (value: string) => {
    if (value === "100") return "100%";
    if (value === "50") return "50%";
    return "0%";
};

// 📌 헬퍼 함수: UI에 표시할 텍스트 변환
const getDisplayValue = (value: string) => {
    if (value === "off" || !value) return "OFF";
    return `${value}%`;
};

export function ResultView({ result }: { result: AnalysisResponse }) {
    const [isRecordingMode, setIsRecordingMode] = useState(false);

    if (isRecordingMode) {
        return (
            <FeedbackRecorder
                targetEffect={result.predicted_effect} // 필요에 따라 display_name 등으로 수정
                onCancel={() => setIsRecordingMode(false)}
            />
        );
    }

    // 1. AI가 반환한 문자열 파싱 (안전하게 try-catch로 감싸기)
    let effectData = null;
    try {
        if (result.predicted_effect) {
            effectData = JSON.parse(result.predicted_effect);
        }
    } catch (error) {
        console.error("이펙터 JSON 파싱 실패:", error);
    }

    return (
        <div className="result-layout">
            <article className="result-main-card" style={{ textAlign: 'center', padding: '3rem' }}>

                {/* 📌 이펙터 결과 시각화 영역 */}
                <div style={{ marginBottom: '3rem' }}>
                    <p className="section-eyebrow">예측된 이펙터 값</p>

                    {!effectData ? (
                        <h2 style={{ fontSize: '1.5rem', color: '#ef4444', margin: '2rem 0' }}>결과를 분석할 수 없습니다.</h2>
                    ) : effectData.clean ? (
                        // 🟢 Clean 모드일 때 (단순 텍스트)
                        <h2 style={{
                            fontSize: '3.5rem',
                            margin: '1.5rem 0',
                            color: '#4ade80',
                            fontWeight: 'bold',
                            letterSpacing: '2px',
                            textTransform: 'uppercase'
                        }}>
                            Clean
                        </h2>
                    ) : (
                        // 🎛️ 이펙터가 적용되었을 때 (막대 그래프 UI)
                        <div style={{ maxWidth: '400px', margin: '2rem auto', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>

                            {/* Distortion 막대 */}
                            <div style={{ textAlign: 'left' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: '#fff', fontWeight: 'bold' }}>
                                    <span>Distortion</span>
                                    <span style={{ color: effectData.dist === 'off' ? '#6b7280' : '#c084fc' }}>
                                        {getDisplayValue(effectData.dist)}
                                    </span>
                                </div>
                                <div style={{ width: '100%', height: '12px', backgroundColor: '#374151', borderRadius: '999px', overflow: 'hidden' }}>
                                    <div style={{ width: getBarWidth(effectData.dist), height: '100%', backgroundColor: '#c084fc', transition: 'width 0.5s ease-out' }} />
                                </div>
                            </div>

                            {/* Delay 막대 */}
                            <div style={{ textAlign: 'left' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: '#fff', fontWeight: 'bold' }}>
                                    <span>Delay</span>
                                    <span style={{ color: effectData.delay === 'off' ? '#6b7280' : '#c084fc' }}>
                                        {getDisplayValue(effectData.delay)}
                                    </span>
                                </div>
                                <div style={{ width: '100%', height: '12px', backgroundColor: '#374151', borderRadius: '999px', overflow: 'hidden' }}>
                                    <div style={{ width: getBarWidth(effectData.delay), height: '100%', backgroundColor: '#c084fc', transition: 'width 0.5s ease-out' }} />
                                </div>
                            </div>

                            {/* Phase 막대 */}
                            <div style={{ textAlign: 'left' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem', color: '#fff', fontWeight: 'bold' }}>
                                    <span>Phase</span>
                                    <span style={{ color: effectData.phase === 'off' ? '#6b7280' : '#c084fc' }}>
                                        {getDisplayValue(effectData.phase)}
                                    </span>
                                </div>
                                <div style={{ width: '100%', height: '12px', backgroundColor: '#374151', borderRadius: '999px', overflow: 'hidden' }}>
                                    <div style={{ width: getBarWidth(effectData.phase), height: '100%', backgroundColor: '#c084fc', transition: 'width 0.5s ease-out' }} />
                                </div>
                            </div>

                        </div>
                    )}
                </div>

                {/* 메타 정보 */}
                <div className="result-meta-pills" style={{ justifyContent: 'center', marginBottom: '2rem' }}>
                    <span>기타 음원: {result.source_name}</span>
                </div>

                {/* 오디오 플레이어 */}
                <div className="result-audio-card">
                    <p className="section-eyebrow" style={{ textAlign: 'left' }}>분리된 기타 소리 들어보기</p>
                    <audio controls src={result.guitar_stem_url} style={{ width: '100%' }}>
                        브라우저가 오디오 재생을 지원하지 않습니다.
                    </audio>
                </div>

                <button
                    className="launch-button"
                    onClick={() => setIsRecordingMode(true)}
                    style={{ marginTop: '2rem' }}
                >
                    내 연주 녹음해서 비교하기
                </button>
            </article>
        </div>
    );
}