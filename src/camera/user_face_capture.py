import cv2

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
