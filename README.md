- 모델이 들어오기 전에 만든 초기 웹캠 0. webcam_pre_yolo.py 

- webcam_detect_*.py 의 결과 사진 : detect_* 에 대응 

# best_cpu_win_123.pt 을 이용해 webcam_detect_1.py, webcam_detect_2.py,  webcam_detect_3.py 
# best_cpu_win_4.pt 를 이용해 webcam_detect_4.py 

1. webcam_detect_1.py 

2. webcam_detect_2.py 

- 탐지된 객체의 바운딩 박스와 라벨 표시

- 클래스별 랜덤 색상

- 객체 중심 좌표(cx, cy) 기준으로 방향 판단 (왼쪽 / 앞쪽 / 오른쪽)

- ‘crosswalk’ 아래쪽 + ‘road’ → 차도로 판단

- 캡처 기능 (s 키 누르면 이미지 저장)

3. webcam_detect_3.py 

- 라벨 겹침 방지: y좌표가 가까운 경우 일정 거리 이상 띄워 배치

4. webcam_detect_4.py 

- 개선된 best.pt 적용, PosixPath 관련 에러 해결
  
- 영상 녹화 기능 (c 키 누르면 녹화시작/종료)




