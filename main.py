import cv2
import application

config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' # 핸드폰 카메라로 객체탐지를 하는 파일
frozen_model = 'frozen_inference_graph.pb'                   # 사람을 탐지할 수 있도록 미리 학습된 모델
model = cv2.dnn_DetectionModel(frozen_model, config_file)  # 데이터셋으로부터 모델 생성.
# 파라미터 : 학습된 가중치(weight)를 포함하는 model, config file : 네트워크 구성이 포함된 텍스트파일

classLabels = []
file_name = 'Lables.txt'                                    #객체 목록 파일

# Labels 파일에 포함된 데이터이름을 불러옴.
with open(file_name, 'rt') as fpt:                          #
    classLabels = fpt.read().rstrip('\n').split('\n')

model.setInputSize(320, 320)  # Set input size for frame. (width, height)
model.setInputScale(1.0 / 127.5)  # Set scalefactor value for frame. (double scale)


model.setInputMean((127.5, 127.5, 127.5))  # 프레임의 평균값을 설정합니다.
model.setInputSwapRB(True)  # Set flag swapRB for frame. (bool swapRB)

cap = cv2.VideoCapture("https://192.168.0.4:8080/video")                                  #0은 연결된 카메라의 인덱스
#cap = cv2.VideoCapture(0)                                                           #VideoCapture 객체를 생성

if cap.isOpened():
    cap = cv2.VideoCapture("https://192.168.0.4:8080/video")               #isOpened()로 에러 확인
#   cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Camera")

font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN  # small size sans-serif font : 글꼴을 설정함.

while True:
    ret, frame = cap.read()       #cap.read()는 재생되는,촬영되는 비디오의 한 프레임씩 읽는다 비디오 프레임을 제대로 읽으면
                                  #ret 값은 True가 되고 실패하면 False가 된다 필요한 경우 ret 값을 체크하여 비디오 프레임을
                                  #제대로 읽었는지 확인 가능 읽은 프레임은 frame변수

    ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
    # confThreshold : 신뢰도를 기준으로 상자를 필터링하는 데 사용되는 임계값입니다.
    # detect() returns (classIds, confidences, boxes)
    # 일단 뭐던간 니가 생각하는게 맞음

    c = 1
    if len(ClassIndex) != 0:

        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
            # zip() : 파라미터들의 데이터를 하나씩 짝지어줌. 여기서는 각각 ClassInd, conf, boxes에 할당됨.
            # flatten : 2차원 배열을 1차원 배열로 수정.

            if ClassInd == 1:
                cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                # img(frame), boxes(왼쪽 위 모서리, 오른쪽 아래 모서리), 색, 두께
                cv2.putText(frame, classLabels[ClassInd - 1] + f'{c}', (boxes[0] + 10, boxes[1] + 40), font,
                            fontScale=font_scale, color=(0, 255, 0), thickness=3)
                c += 1          #사람 수

    cv2.putText(frame, f'Total Persons : {c - 1}', (20, 430), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('object Detection Tutorial', frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
