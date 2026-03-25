import os
import subprocess

def separate_stems_cli(input_audio_path, output_dir="separated_stems"):
    print(f"[{input_audio_path}] 음원 분리를 시작합니다...")
    
    #demuscus를 통해 음원 분리 실행 명령어
    command = [
        #실제 cli 명령어: demucs -n htdemucs -o separated_stems sweet_dreams.mp3
        "demucs", 
        "-n", "htdemucs", 
        "-o", output_dir, 
        input_audio_path
    ]
    
    try:
        subprocess.run(command, check=True)
        print("\n모든 작업이 성공적으로 완료되었습니다!")
        
        #path 설정
        base_name = os.path.splitext(os.path.basename(input_audio_path))[0]
        final_path = os.path.join(output_dir, "htdemucs", base_name)
        print(f"결과물 폴더: {final_path}")
        
    except subprocess.CalledProcessError as e:
        print(f"Demucs 에러 발생: {e}")

if __name__ == "__main__":
    SAMPLE_AUDIO = "test_2.mp3"  # mp3, wav 모두 작동
    
    if os.path.exists(SAMPLE_AUDIO):
        separate_stems_cli(SAMPLE_AUDIO)
    else:
        print(f"'{SAMPLE_AUDIO}' 파일이 없습니다.")