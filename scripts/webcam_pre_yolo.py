import cv2
import numpy as np

# [1] 이미지 전처리 함수
def preprocess(frame, img_size=640):
    image = cv2.resize(frame, (img_size, img_size))
    image = image / 255.0  # 정규화
    image = image.transpose((2, 0, 1))  # HWC → CHW
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

# [2] 후처리 자리를 비워둠
def postprocess(outputs):
    # 나중에 YOLO 예측 결과 해석할 자리
    pass

# [3] 웹캠 루프
def webcam_loop():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open webcam")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # [4] 전처리 적용
        input_tensor = preprocess(frame)

        # [5] (나중에 여기에 모델 추론 들어올 자리)
        # outputs = model.predict(input_tensor)

        # [6] (후처리도 나중에 여기에)
        # result = postprocess(outputs)

        # [7] 현재 프레임 표시
        cv2.imshow("Webcam Preview", frame)

        # [8] 종료 조건
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Webcam closed")
            break

    cap.release()
    cv2.destroyAllWindows()

# 실행
if __name__ == "__main__":
    webcam_loop()
