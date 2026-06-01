import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
import glob
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow import keras
import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication 
import random
import mediapipe as mp



class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(2560, 1440)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.original_pic = QtWidgets.QLabel(self.centralwidget)
        self.original_pic.setGeometry(QtCore.QRect(70, 90, 911, 800))
        self.original_pic.setObjectName("original_pic")

        self.overlay_pic = QtWidgets.QLabel(self.centralwidget)
        self.overlay_pic.setGeometry(QtCore.QRect(1450, 90, 911, 800))
        self.overlay_pic.setObjectName("overlay_pic")

        self.take_pic = QtWidgets.QPushButton(self.centralwidget)
        self.take_pic.setGeometry(QtCore.QRect(70, 950, 531, 131))
        self.take_pic.setObjectName("take_pic")

        self.face_shape = QtWidgets.QPushButton(self.centralwidget)
        self.face_shape.setGeometry(QtCore.QRect(70, 1110, 531, 261))
        self.face_shape.setObjectName("face_shape")

        self.man = QtWidgets.QPushButton(self.centralwidget)
        self.man.setGeometry(QtCore.QRect(1420, 920, 451, 81))
        self.man.setObjectName("man")

        self.women = QtWidgets.QPushButton(self.centralwidget)
        self.women.setGeometry(QtCore.QRect(1910, 920, 451, 81))
        self.women.setObjectName("women")

        self.recommend = QtWidgets.QPushButton(self.centralwidget)
        self.recommend.setGeometry(QtCore.QRect(1420, 1010, 941, 151))
        self.recommend.setObjectName("recommend")

        self.re_recommend = QtWidgets.QPushButton(self.centralwidget)
        self.re_recommend.setGeometry(QtCore.QRect(1420, 1180, 291, 191))
        self.re_recommend.setObjectName("re_recommend")

        self.save = QtWidgets.QPushButton(self.centralwidget)
        self.save.setGeometry(QtCore.QRect(1740, 1180, 271, 191))
        self.save.setObjectName("save")

        self.outout = QtWidgets.QPushButton(self.centralwidget)
        self.outout.setGeometry(QtCore.QRect(2040, 1180, 321, 191))
        self.outout.setObjectName("outout")

        self.face_shape_result = QtWidgets.QLabel(self.centralwidget)
        self.face_shape_result.setGeometry(QtCore.QRect(670, 1100, 421, 261))
        self.face_shape_result.setObjectName("face_shape_result")

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 2560, 22))
        self.menubar.setObjectName("menubar")

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")

        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
        self.man.clicked.connect(self.man_chose)
        self.women.clicked.connect(self.women_chose)
        self.take_pic.clicked.connect(self.capture_pic)
        self.face_shape.clicked.connect(self.face)
        self.recommend.clicked.connect(self.reco)
        self.save.clicked.connect(self.savesave)
        self.outout.clicked.connect(self.outoutout)
        self.re_recommend.clicked.connect(self.rereco)


#ㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡㅡ#
    def capture_pic(self):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("카메라를 열 수 없습니다.")
                return

            print("사진을 찍으려면 'c' 키를 누르세요. 종료하려면 'q' 키를 누르세요")

            cv2.namedWindow('camera', cv2.WINDOW_NORMAL)  # 창 크기 조절 가능하도록 설정
            cv2.resizeWindow('camera', 1280, 720)

            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


            while True:
                ret, frame = cap.read()
                if not ret:
                    print("프레임을 읽을 수 없습니다.")
                    break

                frame = cv2.resize(frame,(1280,720))
                original_frame = frame.copy()


                # 얼굴 감지
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                # 가이드라인 그리기
                guide_color = (0, 255, 0)  # 녹색 가이드라인
                guide_thickness = 2

                guide_w = 660
                guide_h = 550
                guide_x = (1280 - guide_w) // 2
                guide_y = (720 - guide_h) // 2
                            
                cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h), guide_color, guide_thickness)

                face_in_guide = False  # 얼굴이 가이드라인 안에 있는지 확인
                for (x, y, w, h) in faces:
                    if guide_x < x < guide_x + guide_w - w and guide_y < y < guide_y + guide_h - h:
                        face_in_guide = True  # 얼굴이 가이드라인 안에 위치
                        break

                # 얼굴이 가이드라인 안에 들어오면 가이드라인 색상 변경
                if face_in_guide:
                    guide_color = (255, 0, 0)  # 파란색으로 변경
                    cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h), guide_color, guide_thickness)
                else:
                    cv2.rectangle(frame, (guide_x, guide_y), (guide_x + guide_w, guide_y + guide_h), guide_color, guide_thickness)


                cv2.imshow('camera', frame)
                key = cv2.waitKey(1)

                if key == ord('c') and face_in_guide:
                    # 사진 저장 경로 지정 & 가이드라인 안에 얼굴이 있을 때만 촬영
                    save_path = 'F:\\BR\\face_picture.jpg'
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    cv2.imwrite(save_path, original_frame)
                    print(f"사진이 저장되었습니다: {save_path}")

                    # OpenCV BGR 이미지를 RGB로 변환 후 QLabel에 표시
                    image = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                    h, w, ch = image.shape
                    print(f"w : {w}, h : {h}")
                    bytes_per_line = ch * w
                    qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                    
                    pixmap = QPixmap.fromImage(qimg)
                    self.original_pic.setPixmap(pixmap)
                    self.original_pic.setScaledContents(True)  # 이미지가 QLabel 크기에 맞게 조정
                    
                    break
                
                elif key == ord('q'):
                    print("종료합니다.")
                    break

            cap.release()
            cv2.destroyAllWindows()
        
    def face(self):
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
            #img_array = preprocess_image(img_array)  # 훈련 시 사용한 전처리 함수 적용
            #img_array = preprocess_input(img_array)  # InceptionV3의 전처리 함수 사용
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
            self.face_shape_result.setText(f"{face_shape} ({confidence:.2f}%)")
        else:
            print("No photos found in the directory.")

    def man_chose(self):
             # Load saved face image
        image_path = 'F:\\BR\\face_picture.jpg'
        self.hair_path_man = 'F:\\BR\hair_reco\\man'
        if not os.path.exists(image_path):
            print("저장된 사진이 없습니다.")
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print("이미지를 불러올 수 없습니다.")
            return

        # RGB로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        # 얼굴 부분만 segmentation하기 위한 마스크 생성
        mask = np.zeros_like(image)

        # 얼굴 랜드마크가 감지되었으면
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for landmark in face_landmarks.landmark:
                    points.append((int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])))

                # 얼굴 윤곽을 다각형으로 만들어서 마스크로 채우기
                points = np.array(points)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, (255, 255, 255))

        # 얼굴 부분만 추출
        segmented_face = cv2.bitwise_and(image, mask)
        cv2.imwrite('face_picture_seg.png',segmented_face)
        
        # OpenCV BGR 이미지를 RGB로 변환 후 QLabel에 표시
        image = cv2.cvtColor(segmented_face, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.overlay_pic.setPixmap(pixmap)
        self.overlay_pic.setScaledContents(True)  # 이미지가 QLabel 크기에 맞게 조정 

    
    def reco(self):
        
        image_path =  cv2.imwrite('face_picture_seg.png')
        segmented_face = cv2.imread(image_path)

        min_x, min_y, max_x, max_y = self.get_face_bbox(segmented_face)  # 얼굴 영역을 가져오는 함수
        face_width = max_x - min_x
        face_height = max_y - min_y


        if self.man == self.sender():
            hair_path = self.hair_path_man
            print(f"남자 버튼 클릭됨, 헤어 경로: {hair_path}")

        elif self.women == self.sender():  # 버튼을 클릭한 위젯이 self.women인 경우
            hair_path = self.hair_path_woman  # 여자 헤어 경로
            print(f"여자 버튼 클릭됨, 헤어 경로: {hair_path}")

        else:
            print("남자 또는 여자를 선택해 주세요.")

        face_shape = self.face_shape

        available_shapes = os.listdir(hair_path)

        if face_shape in available_shapes:
            # self.face_shape 값이 디렉토리 목록에 있다면 해당 경로로 설정
            selected_hair_path = os.path.join(hair_path, face_shape)
            print(f"선택된 디렉토리 경로: {selected_hair_path}")
        else:
            print(f"{face_shape}에 해당하는 디렉토리가 없습니다.")
            return  # 해당 디렉토리가 없다면 종료

        hair_files = [f for f in os.listdir(selected_hair_path) if f.endswith('.png')]  # PNG 파일만 필터링


        if hair_files:
            selected_hair_file = random.choice(hair_files)  # 랜덤으로 하나 선택
            hair_image_path = os.path.join(selected_hair_path, selected_hair_file)
            print(f"선택된 랜덤 헤어 이미지 경로: {hair_image_path}")

        else:
                print("헤어 이미지가 해당 디렉토리에 없습니다.")


        hair_image = cv2.imread(hair_image_path, cv2.IMREAD_UNCHANGED)

        hair_resized = cv2.resize(hair_image, (face_width, int(face_width * hair_image.shape[0] / hair_image.shape[1])))

            # 헤어 이미지의 알파 채널 (투명도) 추출
        if hair_resized.shape[2] == 4:  # 알파 채널이 있는 경우
            hair_rgb = hair_resized[:, :, :3]
            hair_alpha = hair_resized[:, :, 3]
        else:  # 알파 채널이 없는 경우
            hair_rgb = hair_resized
            hair_alpha = np.ones(hair_resized.shape[:2], dtype=np.uint8) * 255  # 완전 불투명

        # 얼굴의 위쪽에 헤어 이미지 위치 설정
        hair_y_offset = min_y - hair_resized.shape[0] // 3
        hair_x_offset = min_x

        # 헤어 이미지를 얼굴에 오버레이
        overlay = np.zeros_like(segmented_face)
        for c in range(0, 3):  # RGB 채널 처리
            overlay[hair_y_offset:hair_y_offset + hair_rgb.shape[0], hair_x_offset:hair_x_offset + hair_rgb.shape[1], c] = \
                overlay[hair_y_offset:hair_y_offset + hair_rgb.shape[0], hair_x_offset:hair_x_offset + hair_rgb.shape[1], c] * \
                (1 - hair_alpha / 255.0) + hair_rgb[:, :, c] * (hair_alpha / 255.0)

        # 최종 이미지 합성
        final_image = cv2.addWeighted(segmented_face, 1, overlay, 1, 0)

        # 결과를 OpenCV에서 PyQt로 변환하여 QLabel에 표시
        final_image_rgb = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        h, w, ch = final_image_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(final_image_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.overlay_pic.setPixmap(pixmap)
        self.overlay_pic.setScaledContents(True)  # 이미지가 QL

    def rereco(self):
        pass


    def women_chose(self):
        image_path = 'F:\\BR\\face_picture.jpg'
        self.hair_path_women = 'F:\\BR\\hair_reco\\women'

        if not os.path.exists(image_path):
            print("저장된 사진이 없습니다.")
            return

        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)

        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            print("이미지를 불러올 수 없습니다.")
            return

        # RGB로 변환
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        # 얼굴 부분만 segmentation하기 위한 마스크 생성
        mask = np.zeros_like(image)

        # 얼굴 랜드마크가 감지되었으면
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                points = []
                for landmark in face_landmarks.landmark:
                    points.append((int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])))

                # 얼굴 윤곽을 다각형으로 만들어서 마스크로 채우기
                points = np.array(points)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(mask, hull, (255, 255, 255))

        # 얼굴 부분만 추출
        segmented_face = cv2.bitwise_and(image, mask)
        cv2.imwrite('face_picture_seg.png',segmented_face)
        
        # OpenCV BGR 이미지를 RGB로 변환 후 QLabel에 표시
        image = cv2.cvtColor(segmented_face, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)

        pixmap = QPixmap.fromImage(qimg)
        self.overlay_pic.setPixmap(pixmap)
        self.overlay_pic.setScaledContents(True)  # 이미지가 QLabel 크기에 맞게 조정 

    

    def outoutout(self):
        QApplication.quit() 

    
    def savesave(self):
        save_directory =  'overlay_final'

        if self.overlay_pic.pixmap():
            # 저장할 파일 경로
            save_path = os.path.join(save_directory, 'saved_image.png')
            # QPixmap을 PNG 파일로 저장
            self.overlay_pic.pixmap().save(save_path)
            print(f"이미지가 {save_path}에 저장되었습니다.")
        else:
            print("저장할 이미지가 없습니다.")




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.original_pic.setText(_translate("MainWindow", "TextLabel"))
        self.overlay_pic.setText(_translate("MainWindow", "TextLabel"))
        self.take_pic.setText(_translate("MainWindow", "사진촬영"))
        self.face_shape.setText(_translate("MainWindow", "얼굴형 분석"))
        self.man.setText(_translate("MainWindow", "남자 "))
        self.women.setText(_translate("MainWindow", "여자"))
        self.recommend.setText(_translate("MainWindow", "헤어스타일 추천"))
        self.re_recommend.setText(_translate("MainWindow", "재추천"))
        self.save.setText(_translate("MainWindow", "저장"))
        self.outout.setText(_translate("MainWindow", "닫기"))
        self.face_shape_result.setText(_translate("MainWindow", "얼굴형"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())