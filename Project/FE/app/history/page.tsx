"use client";

import { useEffect, useState } from "react";
import { getAllLogs } from "@/lib/api";
import type { AnalysisLog, PredictedEffect } from "@/lib/types";
import { FeedbackRecorder } from "@/components/FeedbackRecorder";

export default function HistoryPage() {
    const [logs, setLogs] = useState<AnalysisLog[]>([]);
    const [selectedLog, setSelectedLog] = useState<AnalysisLog | null>(null);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        const fetchLogs = async () => {
            try {
                const data = await getAllLogs();
                if (data.success) {
                    setLogs(data.logs);
                }
            } catch (error) {
                console.error(error);
            } finally {
                setIsLoading(false);
            }
        };
        fetchLogs();
    }, []);

    // 📌 1. 이름이 없을 경우 파일 경로에서 파일명만 추출하는 헬퍼 함수
    const getDisplayName = (log: AnalysisLog) => {
        if (log.source_info.name && log.source_info.name.trim() !== "") {
            return log.source_info.name;
        }
        if (log.file_path) {
            // 경로에서 마지막 '/' 이후의 문자열(파일명)만 가져옵니다.
            return log.file_path.split('/').pop() || "알 수 없는 오디오";
        }
        return "알 수 없는 오디오";
    };

    // 📌 2. 막대그래프 렌더링용 헬퍼 함수
    const getBarWidth = (val: string | undefined) => {
        if (!val || val === 'off') return '0%';
        return `${val}%`;
    };

    const getDisplayValue = (val: string | undefined) => {
        if (!val || val === 'off') return 'OFF';
        return `${val}%`;
    };

    // 📌 3. Clean 모드(모두 off)인지 판별하는 함수
    const checkIsClean = (effect: PredictedEffect | null) => {
        if (!effect) return true;
        return effect.dist === 'off' && effect.delay === 'off' && effect.phase === 'off';
    };


    // ----------------------------------------------------
    // 🔍 상세 화면 (이펙터 시각화 + 녹음 UI)
    // ----------------------------------------------------
    if (selectedLog) {
        const effectData = selectedLog.predicted_effect;
        const isClean = checkIsClean(effectData);

        return (
            <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto', textAlign: 'center' }}>
                <div style={{ marginBottom: '2rem' }}>
                    <h2 style={{ fontSize: '1.5rem', marginBottom: '0.5rem', textAlign: 'left' }}>
                        {getDisplayName(selectedLog)}
                    </h2>

                    {/* 오디오 미리듣기 */}
                    {selectedLog.file_path && (
                        <div style={{ marginTop: '1.5rem', textAlign: 'left' }}>
                            <p className="section-eyebrow" style={{ marginBottom: '8px' }}>추출된 기타 트랙 원본</p>
                            <audio
                                controls
                                src={`http://127.0.0.1:8000${selectedLog.file_path}`}
                                style={{ width: '100%' }}
                            />
                        </div>
                    )}

                    {/* 📌 요청하신 시각화 UI 블록 적용 */}
                    <div style={{ backgroundColor: '#27272a', padding: '2rem', borderRadius: '12px', marginTop: '1rem' }}>


                        <p style={{ margin: '0 0 1rem 0', color: '#a1a1aa', textAlign: 'left' }}>AI 예측 이펙터 분석 결과</p>

                        {!effectData ? (
                            <h2 style={{ fontSize: '1.5rem', color: '#ef4444', margin: '2rem 0' }}>결과를 분석할 수 없습니다.</h2>
                        ) : isClean ? (
                            // 🟢 Clean 모드일 때
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
                </div>
                <h2>피드백 받기</h2>
                <FeedbackRecorder
                    targetEffect={selectedLog.predicted_effect ? JSON.stringify(selectedLog.predicted_effect) : ""}
                    sourceId={selectedLog.track_id}
                    onCancel={() => setSelectedLog(null)}
                />
            </div>
        );
    }

    // ----------------------------------------------------
    // 📋 리스트 화면
    // ----------------------------------------------------
    return (
        <div style={{ padding: '2rem', maxWidth: '800px', margin: '0 auto' }}>
            <h1 style={{ fontSize: '2rem', marginBottom: '2rem' }}>분석 기록</h1>

            {isLoading ? (
                <p>기록을 불러오는 중...</p>
            ) : logs.length === 0 ? (
                <p style={{ color: '#a1a1aa' }}>아직 분석된 기록이 없습니다.</p>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                    {logs.map((log) => {
                        const isClean = checkIsClean(log.predicted_effect);

                        return (
                            <div
                                key={log.track_id}
                                onClick={() => setSelectedLog(log)}
                                style={{
                                    padding: '1.5rem',
                                    backgroundColor: '#18181b',
                                    border: '1px solid #3f3f46',
                                    borderRadius: '12px',
                                    cursor: 'pointer',
                                    transition: 'background 0.2s',
                                    display: 'flex',
                                    justifyContent: 'space-between',
                                    alignItems: 'center'
                                }}
                                onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#27272a'}
                                onMouseLeave={(e) => e.currentTarget.style.backgroundColor = '#18181b'}
                            >
                                <div>
                                    <h3 style={{ margin: 0, fontSize: '1.2rem', color: 'white' }}>
                                        🎵 {getDisplayName(log)}
                                    </h3>
                                    <p style={{ margin: '0.5rem 0 0 0', color: '#a1a1aa', fontSize: '0.9rem' }}>
                                        클릭해서 확인하기
                                    </p>
                                </div>

                                {/* 리스트 우측에 작게 Clean/Effected 뱃지 표시 */}
                                <div style={{
                                    padding: '0.3rem 0.8rem',
                                    borderRadius: '999px',
                                    fontSize: '0.8rem',
                                    fontWeight: 'bold',
                                    backgroundColor: isClean ? 'rgba(74, 222, 128, 0.1)' : 'rgba(192, 132, 252, 0.1)',
                                    color: isClean ? '#4ade80' : '#c084fc',
                                    border: `1px solid ${isClean ? '#4ade80' : '#c084fc'}`
                                }}>
                                    {isClean ? 'CLEAN' : 'EFFECTER ON'}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}