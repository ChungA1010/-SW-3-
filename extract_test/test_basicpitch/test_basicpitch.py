import os
from basic_pitch.inference import predict_and_save, ICASSP_2022_MODEL_PATH

def convert_audio_to_midi(audio_path, output_directory="midi_results"):
    print(f"[{audio_path}] 풀버전 TF 엔진으로 음표 추출 시작...")
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    try:
        # 풀버전에서는 모델 경로를 억지로 지정할 필요가 없습니다.
        predict_and_save(
            audio_path_list=[audio_path],
            output_directory=output_directory,
            save_midi=True,
            save_model_outputs=False,
            save_notes=False,
            sonify_midi=False,
            model_or_model_path=ICASSP_2022_MODEL_PATH
        )
        print(f"\n오리지널 모델로 MIDI 변환 완료! '{output_directory}' 폴더를 확인하세요.")
        
    except Exception as e:
        print(f"❌ 에러 발생: {e}")

if __name__ == "__main__":
    TARGET_WAV = r"../test_demucs/separated_stems/htdemucs/test_2/other.wav" 
    BASS_WAV = r"../test_demucs/separated_stems/htdemucs/test_2/bass.wav"
    DRUMS_WAV = r"../test_demucs/separated_stems/htdemucs/test_2/drums.wav"
    VOCALS_WAV = r"../test_demucs/separated_stems/htdemucs/test_2/vocals.wav"
    
    PATHS = [TARGET_WAV, BASS_WAV, DRUMS_WAV, VOCALS_WAV]
