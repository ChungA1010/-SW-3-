"use client";

import { useState, useRef } from "react";
import { getFeedback } from "@/lib/api";
import type { FeedbackResponse, AxisFeedback } from "@/lib/types";

interface Props {
    targetEffect?: string;
    sourceId?: number;
    onCancel: () => void;
}

export function FeedbackRecorder({ targetEffect, sourceId, onCancel }: Props) {
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);

    const [feedbackResult, setFeedbackResult] = useState<FeedbackResponse | null>(null);

    const [recordingTime, setRecordingTime] = useState(0);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const animationRef = useRef<number | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);

    const fileInputRef = useRef<HTMLInputElement | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<BlobPart[]>([]);

    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256;
            source.connect(analyser);
            audioContextRef.current = audioContext;

            const mediaRecorder = new MediaRecorder(stream);
            mediaRecorderRef.current = mediaRecorder;
            audioChunksRef.current = [];

            mediaRecorder.ondataavailable = (event) => {
                if (event.data.size > 0) audioChunksRef.current.push(event.data);
            };

            mediaRecorder.onstop = () => {
                const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
                setRecordedBlob(audioBlob);
                stopVisualization();
            };

            mediaRecorder.start();
            setIsRecording(true);
            setRecordingTime(0);
            setFeedbackResult(null);

            timerRef.current = setInterval(() => {
                setRecordingTime(prev => prev + 1);
            }, 1000);

            visualize(analyser);

        } catch (err) {
            console.error("접근 실패:", err);
            alert("마이크 권한을 허용해 주세요!");
        }
    };

    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            setIsRecording(false);
            if (timerRef.current) clearInterval(timerRef.current);
        }
    };

    const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setRecordedBlob(file);
            setFeedbackResult(null);
        }
    };

    const visualize = (analyser: AnalyserNode) => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        const canvasCtx = canvas.getContext("2d");
        if (!canvasCtx) return;

        const bufferLength = analyser.frequencyBinCount;
        const dataArray = new Uint8Array(bufferLength);

        const draw = () => {
            animationRef.current = requestAnimationFrame(draw);
            analyser.getByteFrequencyData(dataArray);

            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, "#f0abfc");
            gradient.addColorStop(1, "#d946ef");

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                canvasCtx.fillStyle = gradient;
                canvasCtx.fillRect(x, canvas.height - barHeight, barWidth, barHeight);
                x += barWidth + 1;
            }
        };
        draw();
    };

    const stopVisualization = () => {
        if (animationRef.current) cancelAnimationFrame(animationRef.current);
        if (audioContextRef.current) audioContextRef.current.close();
    };

    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    };

    const handleSendFeedback = async () => {
        if (!recordedBlob) return;
        if (!sourceId) {
            alert("원본 트랙 정보가 없습니다.");
            return;
        }

        setIsAnalyzing(true);
        setFeedbackResult(null);

        try {
            let fileToSend: File;
            if (recordedBlob instanceof File) {
                fileToSend = recordedBlob;
            } else {
                fileToSend = new File([recordedBlob], "my_feedback.webm", { type: "audio/webm" });
            }

            const response = await getFeedback(fileToSend, sourceId);
            setFeedbackResult(response);
        } catch (error) {
            console.error(error);
            alert("분석 중 오류가 발생했습니다.");
        } finally {
            setIsAnalyzing(false);
        }
    };

    const getScoreColor = (score: number) => {
        if (score >= 80) return '#4ade80';
        if (score >= 65) return '#facc15';
        return '#ef4444';
    };

    const getGradeColor = (grade: string) => {
        if (grade === 'S' || grade === 'A') return '#4ade80';
        if (grade === 'B') return '#facc15';
        return '#ef4444';
    }

    const resetAudio = () => {
        setRecordedBlob(null);
        if (fileInputRef.current) fileInputRef.current.value = "";
    };

    const renderList = (title: string, items: string[], icon: string, color: string) => {
        if (!items || items.length === 0) return null;
        return (
            <div style={{ marginTop: '1rem', backgroundColor: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px' }}>
                <h4 style={{ margin: '0 0 0.5rem 0', color: color, fontSize: '0.95rem', display: 'flex', alignItems: 'center', gap: '6px' }}>
                    {icon} {title}
                </h4>
                <ul style={{ margin: 0, paddingLeft: '1.2rem', color: '#d4d4d8', fontSize: '0.9rem', lineHeight: '1.6' }}>
                    {items.map((item, idx) => <li key={idx} style={{ marginBottom: '4px' }}>{item}</li>)}
                </ul>
            </div>
        );
    };

    return (
        <div className="result-layout">
            <article className="result-main-card">

                {!feedbackResult && (
                    <>
                        <div className={`recorder-timer ${isRecording ? "recording" : ""}`}>
                            {formatTime(recordingTime)}
                        </div>

                        <div className="recorder-visualizer-wrap" style={{ display: isRecording ? 'block' : 'none' }}>
                            <canvas ref={canvasRef} className="recorder-visualizer" width="600" height="100" />
                        </div>

                        {!recordedBlob && (
                            <div style={{ margin: '2.5rem 0', display: 'flex', flexDirection: 'column', alignItems: 'center' }}>
                                {isRecording ? (
                                    <button
                                        className="launch-button"
                                        onClick={stopRecording}
                                        style={{ backgroundColor: '#ff4d4f', color: 'white', width: '100%', maxWidth: '320px' }}
                                    >
                                        🛑 녹음 중지
                                    </button>
                                ) : (
                                    <>
                                        <button className="launch-button" onClick={startRecording} style={{ width: '100%', maxWidth: '320px' }}>
                                            🎙️ 마이크로 직접 녹음
                                        </button>

                                        <div style={{ display: 'flex', alignItems: 'center', width: '100%', maxWidth: '320px', color: '#71717a', fontSize: '0.8rem', margin: '1rem 0' }}>
                                            <div style={{ flex: 1, height: '1px', backgroundColor: '#3f3f46' }}></div>
                                            <span style={{ padding: '0 12px', letterSpacing: '1px', fontWeight: 'bold' }}>OR</span>
                                            <div style={{ flex: 1, height: '1px', backgroundColor: '#3f3f46' }}></div>
                                        </div>

                                        <button
                                            className="launch-button"
                                            onClick={() => fileInputRef.current?.click()}
                                            style={{ backgroundColor: 'rgba(63, 63, 70, 0.3)', border: '1px dashed #52525b', color: 'black', width: '100%', maxWidth: '320px' }}
                                        >
                                            📁 기기내 오디오 파일 업로드
                                        </button>
                                        <input type="file" accept="audio/*" ref={fileInputRef} style={{ display: 'none' }} onChange={handleFileUpload} />
                                    </>
                                )}
                            </div>
                        )}

                        {recordedBlob && !isRecording && (
                            <div style={{ marginTop: '2rem', borderTop: '1px solid var(--line)', paddingTop: '2rem' }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
                                    <p className="section-eyebrow" style={{ margin: 0 }}>업로드된 연주 확인</p>
                                    <button onClick={resetAudio} style={{ background: 'transparent', border: 'none', color: '#f87171', cursor: 'pointer', fontSize: '0.9rem' }}>
                                        다시 선택하기
                                    </button>
                                </div>
                                <audio controls src={URL.createObjectURL(recordedBlob)} style={{ width: '100%', marginBottom: '1.5rem' }} />

                                <button className="launch-button" onClick={handleSendFeedback} disabled={isAnalyzing} style={{ width: '100%' }}>
                                    {isAnalyzing ? "AI가 톤을 분석하고 있습니다..." : "톤 매칭 피드백 받기"}
                                </button>
                            </div>
                        )}
                    </>
                )}

                {/* 📌 피드백 결과 렌더링 영역 */}
                {feedbackResult && feedbackResult.success && feedbackResult.feedback && (
                    <div style={{ textAlign: 'left' }}>
                        <p className="section-eyebrow" style={{ marginBottom: '1.5rem', textAlign: 'center' }}>분석 결과 리포트</p>

                        {/* 1. 이펙터 세팅 피드백 */}
                        <div style={{ marginBottom: '3rem' }}>
                            <h2 style={{ fontSize: '1.2rem', color: '#f4f4f5', borderBottom: '1px solid #3f3f46', paddingBottom: '0.5rem', marginBottom: '1.5rem' }}>
                                🎛️ 이펙터 세팅 분석
                            </h2>
                            <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
                                <p style={{ color: '#a1a1aa', margin: '0 0 0.2rem 0', fontSize: '0.9rem' }}>이펙터 톤 유사도</p>
                                <div style={{ fontSize: '3rem', fontWeight: 'bold', color: getScoreColor(Math.round(feedbackResult.feedback.effect_feedback.overall_similarity * 100)) }}>
                                    {Math.round(feedbackResult.feedback.effect_feedback.overall_similarity * 100)}%
                                </div>
                            </div>

                            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                                {feedbackResult.feedback.effect_feedback.axes.map((axis: AxisFeedback) => (
                                    <div key={axis.axis} style={{
                                        padding: '1.2rem',
                                        backgroundColor: '#27272a',
                                        borderRadius: '12px',
                                        borderLeft: `4px solid ${getScoreColor(Math.round(axis.similarity * 100))}`
                                    }}>
                                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                                            <h3 style={{ textTransform: 'capitalize', margin: 0, color: 'white', fontSize: '1.1rem' }}>
                                                {axis.axis}
                                            </h3>
                                            <span style={{ color: '#a1a1aa', fontSize: '0.9rem' }}>
                                                일치율 {Math.round(axis.similarity * 100)}%
                                            </span>
                                        </div>
                                        <p style={{ margin: 0, color: '#e4e4e7', fontSize: '1rem', lineHeight: '1.5' }}>
                                            {axis.message}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>

                        {/* 2. 연주 품질 피드백 (Tone 제거됨, 시계열만 남음) */}
                        <div style={{ marginBottom: '2rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', borderBottom: '1px solid #3f3f46', paddingBottom: '0.5rem', marginBottom: '1.5rem' }}>
                                <h2 style={{ fontSize: '1.2rem', color: '#f4f4f5', margin: 0 }}>⏱️ 연주 정확도 (Pitch & Rhythm)</h2>
                                <span style={{ fontSize: '1.2rem', fontWeight: 'bold', color: getGradeColor(feedbackResult.feedback.playing_feedback.grade) }}>
                                    {feedbackResult.feedback.playing_feedback.grade} 등급
                                </span>
                            </div>

                            {/* 📌 pitch_reliable 플래그가 false인 경우 보여주는 경고 문구 */}
                            {!feedbackResult.feedback.playing_feedback.pitch_reliable && (
                                <div style={{ marginBottom: '1rem', padding: '0.8rem', borderRadius: '8px', backgroundColor: 'rgba(234, 179, 8, 0.1)', color: '#facc15', fontSize: '0.9rem', border: '1px solid #facc15' }}>
                                    ⚠️ 딜레이/리버브 등 공간계 이펙터가 강하여 음정 분석의 정확도가 다소 떨어질 수 있습니다.
                                </div>
                            )}

                            <div style={{ display: 'flex', justifyContent: 'space-around', backgroundColor: '#27272a', padding: '1.5rem', borderRadius: '12px', marginBottom: '1rem' }}>
                                <div style={{ textAlign: 'center' }}>
                                    <p style={{ margin: '0 0 0.5rem 0', color: '#a1a1aa', fontSize: '0.9rem' }}>종합 점수</p>
                                    <strong style={{ fontSize: '1.8rem', color: getScoreColor(feedbackResult.feedback.playing_feedback.combined_score) }}>
                                        {Math.round(feedbackResult.feedback.playing_feedback.combined_score)}
                                    </strong>
                                </div>
                                <div style={{ width: '1px', backgroundColor: '#3f3f46' }}></div>
                                <div style={{ textAlign: 'center' }}>
                                    <p style={{ margin: '0 0 0.5rem 0', color: '#a1a1aa', fontSize: '0.9rem' }}>음정 (Pitch)</p>
                                    <strong style={{ fontSize: '1.5rem', color: '#e4e4e7' }}>
                                        {Math.round(feedbackResult.feedback.playing_feedback.pitch_score)}
                                    </strong>
                                </div>
                                <div style={{ width: '1px', backgroundColor: '#3f3f46' }}></div>
                                <div style={{ textAlign: 'center' }}>
                                    <p style={{ margin: '0 0 0.5rem 0', color: '#a1a1aa', fontSize: '0.9rem' }}>박자 (Rhythm)</p>
                                    <strong style={{ fontSize: '1.5rem', color: '#e4e4e7' }}>
                                        {Math.round(feedbackResult.feedback.playing_feedback.rhythm_score)}
                                    </strong>
                                </div>
                            </div>

                            {renderList("잘된 점", feedbackResult.feedback.playing_feedback.strengths, "✅", "#4ade80")}
                            {renderList("문제점", feedbackResult.feedback.playing_feedback.issues, "🚨", "#f87171")}
                            {renderList("개선 제안", feedbackResult.feedback.playing_feedback.suggestions, "💡", "#60a5fa")}
                        </div>
                    </div>
                )}

                {/* 📌 에러 화면 */}
                {feedbackResult && !feedbackResult.success && (
                    <div style={{ marginTop: '2rem', padding: '1.5rem', borderRadius: '12px', backgroundColor: 'rgba(239, 68, 68, 0.1)', border: '1px solid #ef4444' }}>
                        <h4 style={{ color: '#ef4444', margin: '0 0 0.5rem 0' }}>❌ 분석 실패</h4>
                        <p style={{ color: 'var(--text)', margin: 0, fontSize: '0.95rem' }}>
                            {feedbackResult.message}
                        </p>
                    </div>
                )}

                <button
                    onClick={onCancel}
                    style={{ marginTop: '2rem', background: 'transparent', color: '#888', border: 'none', cursor: 'pointer', textDecoration: 'underline', width: '100%' }}
                >
                    {feedbackResult ? "다른 연주 분석하러 가기" : "취소"}
                </button>
            </article>
        </div>
    );
}