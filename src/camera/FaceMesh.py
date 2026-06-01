import cv2
import numpy as np
import mediapipe as mp
import os
import glob


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


latest_photo = load_latest_photo("F:\\BR\\utils")

image = cv2.imread(latest_photo)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# FaceMesh 모델 사용
with mp_face_mesh.FaceMesh(static_image_mode=True) as face_mesh:
    results = face_mesh.process(image_rgb)

    if results.multi_face_landmarks:
        # 얼굴 랜드마크를 그립니다
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

            # 랜드마크 포인트를 사용하여 마스크 생성
            height, width, _ = image.shape
            mask = np.zeros((height, width), dtype=np.uint8)

            # 얼굴 랜드마크 좌표
            landmarks = [(int(landmark.x * width), int(landmark.y * height)) for landmark in face_landmarks.landmark]
            
            # 얼굴 랜드마크로 다각형 그리기
            cv2.fillConvexPoly(mask, np.array(landmarks), 255)

            # 결과 마스크를 활용하여 얼굴 부분만 추출
            segmented_face = cv2.bitwise_and(image, image, mask=mask)

# 결과 출력
cv2.imshow('Original Image', image)
cv2.imshow('Segmented Face', segmented_face)
cv2.waitKey(0)
cv2.destroyAllWindows()
