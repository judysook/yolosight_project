import pathlib
pathlib.PosixPath = pathlib.WindowsPath  # 윈도우에서 unidic 호환

import os
import sys
import threading
import time
import random
import glob

import numpy as np
import cv2
import torch
import soundfile as sf
import sounddevice as sd

# ─── 프로젝트 루트 ───────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# ─── TTS 모듈 경로 추가 ────────────────────────────────────
TTS_MODULE_PATH = os.path.join(ROOT, 'tts', 'RealTime_zeroshot_TTS_ko')
sys.path.insert(0, TTS_MODULE_PATH)
# Melotts TTS 모듈 경로도 추가 (tts/Melotts 폴더를 import 경로에 포함)
sys.path.insert(0, os.path.join(ROOT, 'tts', 'Melotts'))

# Zero-shot TTS
from custom_tts import Custom_TTS

# Melotts TTS
from melo.api import TTS as MeloTTS   

# ─── YOLOv5 코드 경로 추가 ─────────────────────────────────
YOLO_PATH = os.path.join(ROOT, 'inference', 'yolov5')
sys.path.insert(0, YOLO_PATH)
from models.common import DetectMultiBackend
from utils.general import non_max_suppression

# ─── 헬퍼 함수 ────────────────────────────────────────────
def find_latest_result_wav():
    """output 폴더에서 가장 최근에 생성된 result_*.wav 반환"""
    dirs = [
        os.path.join(TTS_MODULE_PATH, 'output'),
        os.path.join(os.getcwd(), 'output')
    ]
    files = []
    for d in dirs:
        if os.path.isdir(d):
            files += glob.glob(os.path.join(d, 'result_*.wav'))
    
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def play_audio(res):
    """res가 (buffer, sr)이면 플레이, 아니면 경로로 재생"""
    if isinstance(res, tuple) and len(res)==2:
        buf, sr = res; sd.play(buf, sr); sd.wait()
    elif isinstance(res, str) and os.path.isfile(res):
        data, sr = sf.read(res); sd.play(data, sr); sd.wait()
    else:
        print("[TTS 재생 오류]", res)

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0]/img0_shape[0], img1_shape[1]/img0_shape[1])
        pad  = ((img1_shape[1]-img0_shape[1]*gain)/2,
                (img1_shape[0]-img0_shape[0]*gain)/2)
    else:
        gain, pad = ratio_pad
    coords[:, [0,2]] -= pad[0]
    coords[:, [1,3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

def letterbox(im, new_shape=(640,640), color=(114,114,114),
              auto=False, scaleup=True, stride=32):
    h0, w0 = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0]/h0, new_shape[1]/w0)
    if not scaleup:
        r = min(r, 1.0)
    new_unpad = (int(round(w0*r)), int(round(h0*r)))
    dw, dh = new_shape[1]-new_unpad[0], new_shape[0]-new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    dw /= 2; dh /= 2
    if (w0, h0) != new_unpad[::-1]:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh-0.1)), int(round(dh+0.1))
    left, right = int(round(dw-0.1)), int(round(dw+0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, r, (dw, dh)

# ─── 폴더 준비 ────────────────────────────────────────────
os.makedirs(os.path.join(ROOT,'screenshots'), exist_ok=True)
os.makedirs(os.path.join(ROOT,'recordings'), exist_ok=True)

# ─── 모델 로딩 ────────────────────────────────────────────
WEIGHTS = os.path.join(ROOT,'models','best_cpu_win_45.pt')
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model   = DetectMultiBackend(WEIGHTS, device=device)
stride  = model.stride
names   = model.names
print("클래스:", names)

colors = {i: [random.randint(0,255) for _ in range(3)] for i in range(len(names))}

# ─── TTS 초기화 ────────────────────────────────────────────
tts = Custom_TTS()
tts.set_model()
speaker = os.path.join(TTS_MODULE_PATH,'sample_iena.m4a')
tts.get_reference_speaker(speaker_path=speaker, vad=True)

# ─── 웹캠 오픈 ─────────────────────────────────────────────
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
assert cap.isOpened(), "웹캠을 열 수 없습니다."

recording     = False
out_writer    = None
last_tts_time = 0
min_interval  = 2.0  # 초

# ─── 메인 루프 ────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        break

    img, _, _ = letterbox(frame, 640, stride=stride)
    # BGR→RGB + copy() + CHW
    img = img[:, :, ::-1].copy().transpose(2,0,1)
    img = torch.from_numpy(img).to(device).float()/255.0
    if img.ndimension()==3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.4, 0.45)

    h, w = frame.shape[:2]
    outputs = []
    used_y  = []

    for det in pred:
        if not len(det):
            continue
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
        for *xyxy, conf, cls in det:
            cls = int(cls.item()); label = names[cls]
            x1,y1,x2,y2 = map(int, xyxy)
            cx, cy     = (x1+x2)//2, (y1+y2)//2

            if cx < w/3:      d = '왼쪽'
            elif cx > 2*w/3:  d = '오른쪽'
            else:             d = '앞쪽'

            if label=='crosswalk' and cy>h*0.65:
                text = "차도(횡단보도) 감지됨"
            elif label=='road':
                text = "차도 감지됨"
            else:
                text = f"{d}에 {label} 감지됨"

            outputs.append(text)

            ty = y1 - 10
            while any(abs(ty - yy) < 20 for yy in used_y):
                ty += 20
            used_y.append(ty)

            color = tuple(colors[cls])
            cv2.rectangle(frame, (x1,y1),(x2,y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (x1,ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    if recording and out_writer:
        out_writer.write(frame)

    now = time.time()
    if outputs and now - last_tts_time > min_interval:
        last_tts_time = now

        # output 폴더 비우기
        out_dir = os.path.join(TTS_MODULE_PATH,'output')
        if os.path.isdir(out_dir):
            for f in os.listdir(out_dir):
                os.remove(os.path.join(out_dir,f))

        text = ' '.join(outputs)
        print("TTS ▶", text)

        res = tts.make_speech(text)
        if res is None:
            res = find_latest_result_wav()
            print("▶ fallback to latest wav:", res)

        threading.Thread(target=play_audio, args=(res,), daemon=True).start()

    cv2.imshow('YOLOSight', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        p = os.path.join(ROOT,'screenshots',f"{int(time.time())}.jpg")
        cv2.imwrite(p, frame); print("저장:",p)
    elif key == ord('c'):
        recording = not recording
        if recording:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps    = cap.get(cv2.CAP_PROP_FPS) or 20.0
            out_writer = cv2.VideoWriter(
                os.path.join(ROOT,'recordings',f"{int(time.time())}.mp4"),
                fourcc, fps, (w,h))
            print("녹화 ON")
        else:
            out_writer.release(); out_writer = None
            print("녹화 OFF")
    elif key == ord('q'):
        break

cap.release()
if out_writer: out_writer.release()
cv2.destroyAllWindows()

