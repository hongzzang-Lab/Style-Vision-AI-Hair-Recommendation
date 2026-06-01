import cv2
import numpy as np
import mediapipe as mp
import random
import glob

# MediaPipe FaceMesh 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)

# 헤어 이미지 폴더 경로에서 PNG 파일만 가져오기
hair_images_path = "F:\\BR\\hair_reco\\man\\Square"
hair_images = glob.glob(hair_images_path + "\\*.png")

if not hair_images:
    print("Error: No PNG images found in the specified directory.")
else:
    hair_img_path = random.choice(hair_images)

# 얼굴 이미지 경로
face_img_path = "F:\\BR\\model\\face_picture_seg.png"
face_img = cv2.imread(face_img_path)

# 이마 부분의 랜드마크 추출
def get_forehead_landmarks(landmarks):
    forehead_landmarks = [
        10,  # 랜드마크 인덱스: 좌측 이마
        109, # 우측 이마
        67,  # 이마 상단
        10,  # 추가적인 점들(이마)
    ]
    points = []
    for idx in forehead_landmarks:
        points.append((landmarks[idx].x * face_img.shape[1], landmarks[idx].y * face_img.shape[0]))
    return points

# 오버레이 이미지 처리
def overlay_image(background, overlay, x, y):
    h, w = overlay.shape[:2]
    background_part = background[y:y+h, x:x+w]

    # 그레이스케일인 경우 RGB로 변환
    if len(background_part.shape) == 2:
        background_part = cv2.cvtColor(background_part, cv2.COLOR_GRAY2BGR)
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
    
    mask = overlay[:, :, 0]  # 예시로 첫 번째 채널을 마스크로 사용
    mask = np.expand_dims(mask, axis=-1)  # 마스크를 3채널로 확장
    result = np.where(mask == 0, background_part, overlay)

    background[y:y+h, x:x+w] = result
    return background

# reco 함수 (전체 과정)
def reco():
    # 얼굴 랜드마크 추출
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(face_img_rgb)
    
    if results.multi_face_landmarks:
        # 첫 번째 얼굴만 처리
        landmarks = results.multi_face_landmarks[0].landmark
        
        # 이마 부분 랜드마크 추출
        forehead_points = get_forehead_landmarks(landmarks)
        
        # 헤어 이미지 불러오기
        hair_img = cv2.imread(hair_img_path, cv2.IMREAD_UNCHANGED)  # 알파 채널 포함해서 읽기
        h, w = hair_img.shape[:2]
        
        # 헤어 이미지 크기 조정 (이마 크기에 맞게 조정)
        forehead_width = int(forehead_points[1][0] - forehead_points[0][0])
        forehead_height = int(forehead_points[2][1] - forehead_points[0][1]) // 2
        hair_resized = cv2.resize(hair_img, (forehead_width, forehead_height))
        
        # 헤어 이미지를 이마 부분에 오버레이
        x, y = int(forehead_points[0][0]), int(forehead_points[0][1])
        result_img = overlay_image(face_img, hair_resized, x, y)
        
        # 결과 이미지 띄우기 (overlay_pic에 표시)
        cv2.imshow("Overlay Image", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# reco 함수 호출
reco()
