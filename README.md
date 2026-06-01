
---

## Style-Vision-AI-Hair-Recommendation

```md
# StyleVision: AI-Powered Hairstyle Recommendation

사용자의 얼굴을 촬영하고, 얼굴형을 분석한 뒤 어울리는 헤어스타일을 추천하는 AI 기반 헤어스타일 추천 시스템입니다.

## Project Overview

StyleVision은 OpenCV, TensorFlow/Keras, MediaPipe FaceMesh, PyQt5를 활용한 데스크톱 GUI 프로젝트입니다.  
카메라로 사용자의 얼굴 이미지를 촬영하고, InceptionV3 기반 얼굴형 분류 모델을 통해 얼굴형을 예측합니다.  
이후 MediaPipe FaceMesh로 얼굴 영역을 추출하고, 성별 및 얼굴형에 맞는 헤어스타일 이미지를 선택하여 얼굴 이미지 위에 합성하는 것을 목표로 합니다.

## Features

- PyQt5 기반 데스크톱 GUI
- OpenCV 기반 얼굴 사진 촬영
- 얼굴 가이드라인 기반 촬영 보조
- InceptionV3 기반 얼굴형 분류
- 얼굴형 클래스 분류: Heart, Oblong, Oval, Round, Square
- MediaPipe FaceMesh 기반 얼굴 영역 추출
- 성별 및 얼굴형 기반 헤어스타일 추천
- 헤어 PNG 이미지 오버레이
- 추천 결과 저장 기능

## Face Shape Classes

```text
Heart
Oblong
Oval
Round
Square
