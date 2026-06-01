import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel

# 1. PyQt 애플리케이션 생성
app = QApplication(sys.argv)  # 애플리케이션 객체 생성

# 2. 메인 윈도우 설정
window = QWidget()  # 메인 윈도우 위젯 생성
window.setWindowTitle('PyQt5 애플리케이션')  # 윈도우 제목 설정
window.setGeometry(100, 100, 300, 200)  # 윈도우 위치와 크기 설정

# 3. 간단한 라벨 추가
label = QLabel('PyQt5 실행 성공!', window)  # 라벨 생성 후 윈도우에 추가
label.move(100, 90)  # 라벨 위치 지정

# 4. 메인 윈도우 표시
window.show()  # 윈도우를 화면에 표시

# 5. 애플리케이션 실행
sys.exit(app.exec_())  # 애플리케이션 이벤트 루프 실행

# 터미널에 designer 명령어 ---> pyqt 실행!
