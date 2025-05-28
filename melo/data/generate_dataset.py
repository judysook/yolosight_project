# generate_dataset.py
#오류때문에 일단 사용안함XXX
import os
from TTS.api import TTS

# ✅ 키워드 + 실제 문장 매핑
data_items = {
    "kickboard": "킥보드",
    "bicycle": "자전거",
    "children": "아이들",
    "dog": "강아지",
    "construction": "공사중",
    "manhole": "맨홀",
    "crosswalk": "횡단보도",
    "bike": "자전거",
    "stairs": "계단",
    "car": "자동차"
}

# ✅ 폴더 생성
os.makedirs("data", exist_ok=True)

# ✅ metadata.list 작성
with open("data/metadata.list", "w", encoding="utf-8") as f:
    for fname, keyword in data_items.items():
        f.write(f"{fname}|{keyword}\n")

# ✅ TTS 음성 생성
# 안정적으로 작동하는 한국어 모델 사용
tts = TTS(model_name="tts_models/ko/kaist/tacotron2", progress_bar=False)

for fname, keyword in data_items.items():
    sentence = f"전방에 {keyword}가 있습니다"
    file_path = f"data/{fname}.wav"
    tts.tts_to_file(text=sentence, file_path=file_path)
    print(f"✅ 생성됨: {file_path} ← {sentence}")