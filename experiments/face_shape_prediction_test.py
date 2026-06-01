import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input


# 모델 로드
model = load_model('weights/inceptionv3_model_weights')

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss={'main_output': 'categorical_crossentropy', 'auxiliary_output': 'categorical_crossentropy'},
    metrics={'main_output': 'accuracy', 'auxiliary_output': 'accuracy'}
)

class_names = ['Heart', 'Oblong', 'Oval', 'Round', 'Square']

# 훈련 시 사용한 전처리 함수
def preprocess_image(image):
    if image.shape[-1] == 1:  # 그레이스케일 이미지 확인
        image = tf.image.grayscale_to_rgb(image)  # 그레이스케일 -> RGB 변환
    return image

def preprocess_and_predict(image_path):
    # 새 이미지 불러오기 및 전처리
    img = image.load_img(image_path, target_size=(299, 299))  # 모델에 맞는 크기로 조정
    img_array = image.img_to_array(img)
    img_array = preprocess_image(img_array)  # 훈련 시 사용한 전처리 함수 적용
    img_array = preprocess_input(img_array)  # InceptionV3의 전처리 함수 사용
    img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가

    # 예측 실행
    predictions = model.predict(img_array)
    print(predictions)
    main_output = predictions[0]  # 주 출력(main_output) 사용 (리스트 형식일 때)

    # 클래스 예측 및 확률 추출
    predicted_class = np.argmax(main_output, axis=1)[0]
    confidence = np.max(main_output, axis=1)[0] * 100  # 확률 백분율로 변환
    class_label = class_names[predicted_class]
    
    return class_label, confidence

def load_latest_photo(directory):
    files = glob.glob(os.path.join(directory, '*.jpg'))  # 또는 '*.png'로 변경 가능
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return latest_file


latest_photo = load_latest_photo("F:\\BR")

if latest_photo:
    # 예측 결과 출력
    face_shape, confidence = preprocess_and_predict(latest_photo)
    print(f"The predicted face shape is '{face_shape}' with a confidence of {confidence:.2f}%.")
else:
    print("No photos found in the directory.")
    