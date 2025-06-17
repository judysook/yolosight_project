# 객체가 겹치면 레이블 위치를 조정하는 코드 포함 

import numpy as np
import cv2
import torch
from pathlib import Path
from models.common import DetectMultiBackend
from utils.general import non_max_suppression
import random
import time

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, :4] = coords[:, :4].clamp(min=0)
    return coords

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=False,
              scaleFill=False, scaleup=True, stride=32):
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return im, r, (dw, dh)


# 모델 로딩
weights = 'best_cpu_win.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DetectMultiBackend(weights, device=device)
stride = model.stride
names = model.names
print(names)  # 전체 클래스 출력

# 클래스별 색상 지정
colors = {i: [random.randint(0, 255) for _ in range(3)] for i in range(len(names))}

# 웹캠 열기
cap = cv2.VideoCapture(0)
assert cap.isOpened(), '웹캠을 열 수 없습니다.'

while True:
    ret, frame = cap.read()
    if not ret:
        break
    used_label_y = []
    used_label_x = []

    img = letterbox(frame, 640, stride=stride)[0]
    img = img.transpose((2, 0, 1))[::-1].copy()  # HWC to CHW, BGR to RGB
    img = torch.from_numpy(img).to(device).float()
    img /= 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)
    pred = non_max_suppression(pred, 0.4, 0.45)

    h, w, _ = frame.shape
    direction_output = []

    for det in pred:
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], frame.shape).round()
            for *xyxy, conf, cls in det:
                cls = int(cls.item())
                label = f'{names[cls]} {conf:.2f}'
                xyxy = [int(x.item()) for x in xyxy]

                cx = (xyxy[0] + xyxy[2]) // 2
                cy = (xyxy[1] + xyxy[3]) // 2

                # 방향 판단
                if cx < w // 3:
                    direction = "왼쪽"
                elif cx > w * 2 // 3:
                    direction = "오른쪽"
                else:
                    direction = "앞쪽"

                # 차도 판단
                if names[cls] == 'crosswalk' and cy > h * 0.65:
                    direction_output.append("차도 (횡단보도)")
                elif names[cls] == 'road':
                    direction_output.append("차도")
                else:
                    direction_output.append(f"{direction}에 {names[cls]} 감지됨")

                # ===== 라벨 겹침 방지 =====
                label_x = xyxy[0]
                label_y = xyxy[1] - 10
                while any(abs(label_y - used_y) < 20 for used_y in used_label_y):
                    label_y += 20
                used_label_y.append(label_y)

                # 바운딩 박스와 라벨 출력
                cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), colors[cls].tolist(), 2)
                cv2.putText(frame, label, (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[cls].tolist(), 2)

    for msg in direction_output:
        print(msg)  # 🔈 TTS 연동 지점

    cv2.imshow('YOLOv5 Webcam Detection', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        cv2.imwrite(f'detection_{int(time.time())}.jpg', frame)
        print("이미지 저장됨!")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()