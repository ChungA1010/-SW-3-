import os
from music21 import converter, instrument, tempo, meter

def generate_sheet_music(midi_file, output_xml="my_sheet"):
    print(f"🎼 [{midi_file}] 파이썬으로 악보 변환을 시작합니다...")

    try:
        # 1. MIDI 데이터 불러오기
        score = converter.parse(midi_file)
        
        if output_xml == "guitar":
            inst = instrument.ElectricGuitar()
        elif output_xml == "bass":
            inst = instrument.ElectricBass()
        elif output_xml == "drums":
            inst = instrument.Percussion()
        elif output_xml == "vocals":
            inst = instrument.Vocalist()

        for part in score.parts:
            part.replace(part.getInstrument(returnDefault=True), inst)
        
        score.insert(0, tempo.MetronomeMark(number=120))
        score.insert(0, meter.TimeSignature('4/4'))

        # 2. 박자 정렬 (Quantize) - AI 악보 생성의 핵심!
        # 이를 사람이 읽기 편한 8분음표, 16분음표 단위로 반올림
        print("사람이 읽을 수 있도록 박자를 깔끔하게 정리하는 중...")
        
        score.quantize(
            quarterLengthDivisors=[4, 8],  # 16분, 32분 느낌
            processOffsets=True,
            processDurations=True,
            inPlace=True
        )
        
        
        score.makeMeasures(inPlace=True)
        score.makeNotation(inPlace=True)
        
        for n in score.recurse().notes:
            if n.duration.quarterLength < 0.25:  # 16분음표보다 짧으면 제거
                n.activeSite.remove(n)

        # 3. MusicXML (디지털 악보 표준 파일) 형태로 저장
        print("기타 악보 파일(.xml)을 생성하는 중...")
        score.write('musicxml', fp=f"{output_xml}.xml")

        print(f"\n 악보 파일 생성 완료! [{output_xml}]")
        print("--------------------------------------------------")
        print("팁: 웹 뷰어/MuseScore에 넣으면")
        print("    오선지 악보가 바로 나타납니다!")

    except Exception as e:
        print(f"악보 변환 중 에러 발생: {e}")

if __name__ == "__main__":
    # 🚨 앞서 Basic Pitch로 성공적으로 뽑아낸 MIDI 파일의 정확한 경로를 넣어주세요.
    # 예시: r"D:\extract_test\midi_results\other_basic_pitch.mid"
    GUITAR_MIDI = r"../test_basicpitch/midi_results/other_basic_pitch.mid"
    BASS_MIDI = r"../test_basicpitch/midi_results/bass_basic_pitch.mid"
    DRUMS_MIDI = r"../test_basicpitch/midi_results/drums_basic_pitch.mid"
    VOCALS_MIDI = r"../test_basicpitch/midi_results/vocals_basic_pitch.mid" 
    
    if os.path.exists(GUITAR_MIDI):
        generate_sheet_music(GUITAR_MIDI,"guitar")
    else:
        print(f"파일을 찾을 수 없습니다: {GUITAR_MIDI}")
        print("파일 경로가 정확한지 다시 한번 확인해 주세요!")
        
    if os.path.exists(BASS_MIDI):
        generate_sheet_music(BASS_MIDI,"bass")
    else:
        print(f"파일을 찾을 수 없습니다: {BASS_MIDI}")
        print("파일 경로가 정확한지 다시 한번 확인해 주세요!")
        
    if os.path.exists(DRUMS_MIDI):
        generate_sheet_music(DRUMS_MIDI,"drums")
    else:
        print(f"파일을 찾을 수 없습니다: {DRUMS_MIDI}")
        print("파일 경로가 정확한지 다시 한번 확인해 주세요!")
        
    if os.path.exists(VOCALS_MIDI):
        generate_sheet_music(VOCALS_MIDI,"vocals")
    else:
        print(f"파일을 찾을 수 없습니다: {VOCALS_MIDI}")
        print("파일 경로가 정확한지 다시 한번 확인해 주세요!")
        
