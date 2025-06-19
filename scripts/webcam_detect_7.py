import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # unidic 윈도우 호환용

import os
import sys
sys.path.insert(0, TTS_MODULE_PATH)
from custom_tts import Custom_TTS
import soundfile as sf, sounddevice as sd
import threading
import time
import random
sample_speaker = os.path.join(TTS_MODULE_PATH, 'sample_iena.m4a')
print("샘플 화자 임베딩:", sample_speaker)
tts_module.get_reference_speaker(speaker_path=sample_speaker, vad=True)


# ─── 프로젝트 루트 경로 계산 ───
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ─── RealTime Zero-shot TTS 모듈 경로 추가 ───
TTS_MODULE_PATH = os.path.join(ROOT, 'tts', 'RealTime_zeroshot_TTS_ko')
sys.path.insert(0, TTS_MODULE_PATH)
from custom_tts import Custom_TTS  # 제로샷 TTS
import soundfile as sf
import sounddevice as sd

# ─── YOLOv5 코드 경로 추가 ───
YOLO_PATH = os.path.join(ROOT, 'inference', 'yolov5')
sys.path.insert(0, YOLO_PATH)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

import numpy as np
import cv2
import torch
from pathlib import Path


# ─── 유틸 함수 정의 ───────────────────────────
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # h,w
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    # scale
    r = min(new_shape[0]/shape[0], new_shape[1]/shape[1])
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(shape[1]*r)), int(round(shape[0]*r)))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

def play_async(wav_path):
    data, sr = sf.read(wav_path)
    sd.play(data, sr)

# ─── 디렉터리 자동 생성 ─────────────────────────
os.makedirs(os.path.join(ROOT, 'screenshots'), exist_ok=True)
os.makedirs(os.path.join(ROOT, 'recordings'), exist_ok=True)

# ─── 모델 로딩 ───────────────────────────────────
WEIGHTS_PATH = os.path.join(ROOT, 'models', 'best_cpu_win_45.pt')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = DetectMultiBackend(WEIGHTS_PATH, device=device)
stride = model.stride
names  = model.names
print("Classes:", names)

# 랜덤 색상
colors = {i: [random.randint(0,255) for _ in range(3)] for i in range(len(names))}

tts_module = Custom_TTS()
tts_module.set_model()

sample_path = os.path.join(TTS_MODULE_PATH, 'sample_iena.m4a')
tts_module.get_reference_speaker(speaker_path=sample_path, vad=True)
speaker_path=sample_speaker,
vad=True

# ─── 웹캠 열기 ───────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), '웹캠을 열 수 없습니다.'

# ─── Zero-shot TTS 초기화 ──────────────────────────
tts_module    = Custom_TTS()
tts_module.set_model()

# ─── 상태 변수 ───────────────────────────────────
recording     = False
out           = None
last_tts_time = 0
min_interval  = 2.0  # TTS 최소 재생 간격(초)

# ─── 메인 루프 ───────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 전처리
    img, ratio, pad = letterbox(frame, 640, stride=stride)
    img = img.transpose((2,0,1))[::-1].copy()  # BGR→RGB, HWC→CHW
    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension()==3:
        img = img.unsqueeze(0)

    # 추론 및 NMS
    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.4, 0.45)

    h, w = frame.shape[:2]
    direction_output = []
    used_label_y    = []

    # 검출된 객체 처리
    for det in pred:
        if not len(det):
            continue
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in det:
            cls   = int(cls.item())
            label = names[cls]
            xyxy  = [int(x.item()) for x in xyxy]
            x1,y1,x2,y2 = xyxy
            cx, cy = (x1+x2)//2, (y1+y2)//2

            # 방향 판단
            if   cx < w//3:   direction = '왼쪽'
            elif cx > w*2//3: direction = '오른쪽'
            else:             direction = '앞쪽'

            # 차도 판단
            if label=='crosswalk' and cy > h*0.65:
                direction_output.append("차도 (횡단보도)")
            elif label=='road':
                direction_output.append("차도")
            else:
                direction_output.append(f"{direction}에 {label} 감지됨")

            # 라벨 겹침 방지
            ly = y1 - 10
            while any(abs(ly - uy) < 20 for uy in used_label_y):
                ly += 20
            used_label_y.append(ly)

            # 박스 그리기
            color = tuple(colors[cls])
            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, f'{label} {conf:.2f}', (x1, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # 스크린샷/녹화
    if recording and out is not None:
        out.write(frame)

    # TTS 처리
    sentences = direction_output  # 간단히 재활용
    now = time.time()
    if sentences and now - last_tts_time > min_interval:
        last_tts_time = now
        for sent in sentences:
            print("TTS ▶", sent)
            wav_path = tts_module.make_speech(sent)
            threading.Thread(target=play_async, args=(wav_path,), daemon=True).start()

    # 영상 출력 & 키 입력
    cv2.imshow('YOLOSight Webcam Detection', frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        fname = os.path.join(ROOT, 'screenshots', f'detect_{int(time.time())}.jpg')
        cv2.imwrite(fname, frame)
        print("스크린샷 저장:", fname)

    elif key == ord('c'):
        recording = not recording
        if recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
            out    = cv2.VideoWriter(
                os.path.join(ROOT, 'recordings', f'record_{int(time.time())}.mp4'),
                fourcc, fps, (w, h)
            )
            print("녹화 시작")
        else:
            out.release()
            out = None
            print("녹화 종료")

    elif key == ord('q'):
        break

# 종료
cap.release()
if out:
    out.release()
cv2.destroyAllWindows()
