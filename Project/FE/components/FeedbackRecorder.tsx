"use client";

import { useState, useRef } from "react";
import { getFeedback } from "@/lib/api";
import { SimpleResponse } from "@/lib/types";

interface Props {
    targetEffect?: string;
    sourceId?: number; // 👈 📌 ResultView에서 받아올 Props 정의
    onCancel: () => void;
}

export function FeedbackRecorder({ targetEffect, sourceId, onCancel }: Props) {
    const [isRecording, setIsRecording] = useState(false);
    const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
    const [isAnalyzing, setIsAnalyzing] = useState(false);
    const [feedbackResult, setFeedbackResult] = useState<SimpleResponse | null>(null);

    // ⏱️ 타이머 및 시각화 관련 상태
    const [recordingTime, setRecordingTime] = useState(0);
    const timerRef = useRef<NodeJS.Timeout | null>(null);
    const canvasRef = useRef<HTMLCanvasElement | null>(null);
    const animationRef = useRef<number | null>(null);
    const audioContextRef = useRef<AudioContext | null>(null);

    // 녹음 데이터 저장, useRef 사용해 rerender 방지
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const audioChunksRef = useRef<BlobPart[]>([]);


    //녹음 시작
    const startRecording = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // 1. Web Audio API 설정 (시각화용)
            const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
            const source = audioContext.createMediaStreamSource(stream);
            const analyser = audioContext.createAnalyser();
            analyser.fftSize = 256; // 파형의 정밀도
            source.connect(analyser);
            audioContextRef.current = audioContext;

            // 2. 미디어 레코더 설정
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

            // 3. 타이머 및 시각화 시작
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



    //녹음 중지
    const stopRecording = () => {
        if (mediaRecorderRef.current && isRecording) {
            mediaRecorderRef.current.stop();
            mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
            setIsRecording(false);
            if (timerRef.current) clearInterval(timerRef.current);
        }
    };


    //실시간 파형 그리기 로직
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

            // 캔버스 초기화
            canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

            const barWidth = (canvas.width / bufferLength) * 2.5;
            let barHeight;
            let x = 0;

            // 📌 테마에 맞춘 그라데이션 색상 생성 (위: 핑크, 아래: 마젠타)
            const gradient = canvasCtx.createLinearGradient(0, 0, 0, canvas.height);
            gradient.addColorStop(0, "#f0abfc");
            gradient.addColorStop(1, "#d946ef"); // var(--accent)

            for (let i = 0; i < bufferLength; i++) {
                barHeight = dataArray[i] / 2;
                canvasCtx.fillStyle = gradient; // 그라데이션 적용
                // 막대 끝을 둥글게 보이기 위해 약간의 y축 오프셋을 줄 수도 있습니다.
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

    // 시간 포맷팅 (00:00)
    const formatTime = (seconds: number) => {
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        return `${mins.toString().padStart(2, "0")}:${secs.toString().padStart(2, "0")}`;
    };

    //백엔드로 전달할 수 있게 blob을 file 형식으로 변환
    const handleSendFeedback = async () => {
        //녹음된 파일이 없으면 실행하지 않음
        if (!recordedBlob) return;
        if (!sourceId) {
            alert("원본 트랙 정보가 없습니다.");
            return;
        }

        setIsAnalyzing(true);
        setFeedbackResult(null);

        try {
            //여기서 Blob을 File 객체로 변환합니다.
            const file = new File([recordedBlob], "my_feedback.webm", { type: "audio/webm" });

            //변환된 file을 getFeedback에 전달합니다!
            const response = await getFeedback(file, sourceId);
            setFeedbackResult(response);
        } catch (error) {
            console.error(error);
            alert("분석 중 오류가 발생했습니다.");
        } finally {
            setIsAnalyzing(false);
        }
    };


    return (
        <div className="result-layout">
            <article className="result-main-card">
                {/*타이머 표시 */}
                <div className={`recorder-timer ${isRecording ? "recording" : ""}`}>
                    {formatTime(recordingTime)}
                </div>

                <div className="recorder-visualizer-wrap" style={{ display: isRecording ? 'block' : 'none' }}>
                    <canvas
                        ref={canvasRef}
                        className="recorder-visualizer"
                        width="600" // 해상도를 높여서 더 선명하게 (CSS가 크기 조절함)
                        height="100"
                    />
                </div>

                <div style={{ margin: '2rem 0', display: 'flex', gap: '1rem', justifyContent: 'center' }}>
                    {isRecording ? (
                        <button
                            className="launch-button"
                            onClick={stopRecording}
                            style={{ backgroundColor: '#ff4d4f', color: 'white' }}
                        >
                            녹음 중지
                        </button>
                    ) : (
                        <button className="launch-button" onClick={startRecording}>
                            녹음 시작
                        </button>
                    )}
                </div>

                {/*피드백 받기 버튼*/}
                {recordedBlob && !isRecording && (
                    <div style={{ marginTop: '2rem', borderTop: '1px solid var(--line)', paddingTop: '2rem' }}>
                        <p className="section-eyebrow" style={{ textAlign: 'left', marginBottom: '12px' }}>녹음된 연주 확인</p>
                        <audio controls src={URL.createObjectURL(recordedBlob)} style={{ width: '100%', marginBottom: '1.5rem' }} />

                        <button
                            className="launch-button"
                            onClick={handleSendFeedback}
                            disabled={isAnalyzing}
                            style={{ width: '100%' }}
                        >
                            {isAnalyzing ? "분석 중..." : "피드백 받기"}
                        </button>
                    </div>
                )}

                {/*피드백 결과 UI (SimpleResponse)*/}
                {feedbackResult && (

                    <div style={{
                        marginTop: '2rem',
                        padding: '1.5rem',
                        borderRadius: '12px',
                        backgroundColor: feedbackResult.success ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        border: `1px solid ${feedbackResult.success ? '#10b981' : '#ef4444'}`
                    }}>
                        <h4 style={{ color: feedbackResult.success ? '#10b981' : '#ef4444', margin: '0 0 0.5rem 0' }}>
                            {feedbackResult.success ? "✅ 전송 성공!" : "❌ 전송 실패"}
                        </h4>
                        <p style={{ color: 'var(--text)', margin: 0, fontSize: '0.95rem' }}>
                            {feedbackResult.message || (feedbackResult.success ? "요청이 성공적으로 처리되었습니다." : "알 수 없는 오류가 발생했습니다.")}
                        </p>
                    </div>

                )}

                <button
                    onClick={onCancel}
                    style={{ marginTop: '2rem', background: 'transparent', color: '#888', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
                >
                    취소
                </button>
            </article>
        </div>
    );
}