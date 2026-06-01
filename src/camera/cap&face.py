import cv2
import numpy as np
import mediapipe as mp
import os
import glob


# 카메라 열기
cap = cv2.VideoCapture(0)  # 0은 기본 카메라를 나타냅니다.

if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
    exit()

print("사진을 찍으려면 'c' 키를 누르세요. 종료하려면 'q' 키를 누르세요.")

while True:
    # 카메라로부터 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽을 수 없습니다.")
        break

    # 현재 프레임 보여주기
    cv2.imshow('Camera', frame)

    # 키 입력 대기
    key = cv2.waitKey(1)

    # 'c' 키가 눌리면 사진 저장
    if key == ord('c'):
        cv2.imwrite('user_face.jpg', frame)
        print("사진이 저장되었습니다: user_face.jpg")

    # 'q' 키가 눌리면 종료
    elif key == ord('q'):
        print("종료합니다.")
        break

# 카메라 닫기
cap.release()
cv2.destroyAllWindows()
#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ촬영 후ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
# MediaPipe 초기화

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# 이미지 로드

def load_latest_photo(directory):
    files = glob.glob(os.path.join(directory, '*.jpg'))  # 또는 '*.png'로 변경 가능
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file


latest_photo = load_latest_photo("F:\BR")

image = cv2.imread(latest_photo)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 원본 이미지의 복사본 생성
image_copy = image.copy()

# FaceMesh 모델 사용
image_copy = image.copy()

with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        height, width, _ = image_copy.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        for face_landmarks in results.multi_face_landmarks:
            # 얼굴 랜드마크 좌표
            landmarks = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark]

            # 얼굴 랜드마크로 다각형 그리기 (최소 3개의 점이 있을 때만)
            if len(landmarks) >= 3:
                cv2.fillConvexPoly(mask, np.array(landmarks), 255)

                # 결과 마스크를 활용하여 얼굴 부분만 추출
                segmented_face = cv2.bitwise_and(image_copy, image_copy, mask=mask)

                # 얼굴 부분만 자르기
                # 얼굴 영역 바운딩 박스 구하기
                
                x, y, w, h = cv2.boundingRect(np.array(landmarks))

                # 원본 이미지에서 얼굴 영역을 잘라내기
                face_only = image_copy[y:y+h, x:x+w].copy()

                 # 잘라낸 얼굴 부분에 마스크 적용하기
                mask_face = mask[y:y+h, x:x+w]
                face_only = cv2.bitwise_and(face_only, face_only, mask=mask_face)

# 결과 출력
cv2.imshow('Segmented Face', face_only)


cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()


