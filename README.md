# 🚦 AI 기반 객체 탐지 및 위험 분석 시스템

> YOLOv8과 U-Net 기반 실시간 위험 감지 시스템
> 세종시 자율주행 AI 경진대회 출품 프로젝트

---

# 📌 Overview

CCTV 영상 기반으로 보행자와 이동체 간 충돌 위험을 분석하고,
위험 상황 발생 시 실시간으로 경고를 제공하는 AI 기반 시스템입니다.

본 프로젝트에서는 객체 탐지(Object Detection)와
세그멘테이션(Semantic Segmentation)을 결합하여
횡단보도 주변 위험 상황을 분석할 수 있는 시스템을 구현했습니다.

---

# 📅 Project Information

| Category     | Content                     |
| ------------ | --------------------------- |
| Project Name | AI 기반 객체 탐지 및 위험 분석 기술 개발   |
| Period       | 2024.01 ~ 2024.02           |
| Team         | 박진성, 김성연                    |
| Goal         | 실시간 위험 감지 및 충돌 위험 분석 시스템 개발 |

---

# 🛠 Tech Stack

## AI / Vision

* YOLOv8
* U-Net
* OpenCV
* PyTorch

## Language

* Python

## Development Process

* Agile
* V-Model

---

# 🎯 Development Goal

* 세종시 자율주행 AI 경진대회 출품용 시스템 개발
* CCTV 기반 보행자·이동체 충돌 위험 분석
* 실시간 위험 감지 및 경고 시스템 구현
* 객체 탐지 + 세그멘테이션 융합 기반 위험 판단 기술 개발

---

# ⚙️ System Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/1f5f0f9a-4e84-4605-a18f-08d22688a06d" width="700"/>
</p>

---

# 🔍 Key Features

## 🚶 횡단보도 영역 세그멘테이션

U-Net 모델을 활용하여 CCTV 영상 내 횡단보도 영역을 분리하고
위험 판단에 필요한 도로 영역 정보를 추출했습니다.

---

## 🚲 이동체 객체 탐지

YOLOv8 Fine-Tuning을 통해 다음 클래스를 추가 학습했습니다.

* Bicycle-Human
* Kickboard-Human

이를 통해 일반 보행자뿐 아니라 이동체 탑승 객체까지 탐지할 수 있도록 개선했습니다.

---

## ⚠️ 위험 분석 알고리즘 설계

객체 위치 좌표와 횡단보도 영역 좌표를 비교하여
충돌 위험 여부를 판단하는 알고리즘을 구현했습니다.

또한 YOLOv8 기반 3D 공간 인식을 적용하여
위험 탐지 정확도를 향상시켰습니다.

---

# 🧪 Testing

프로젝트 품질 향상을 위해 V-모델 기반 테스트를 수행했습니다.

* Unit Test
* Component Test
* System Test

테스트 과정에서 객체 인식 오류 및 위험 판단 로직을 지속적으로 개선했습니다.

---

# 📷 Result

## 위험 객체 탐지 결과

<p align="center">
  <img src="https://github.com/user-attachments/assets/b21c0346-dc85-410e-b7dc-3f5da155941f" width="700"/>
</p>

<p align="center">
  <img src="https://github.com/user-attachments/assets/be9952f6-cb90-43e6-bd38-630ebea6e413" width="700"/>
</p>

---

# 🏆 Achievement

* 🥉 세종시 자율주행 AI 경진대회 3등 수상
* 🏅 KIIECT 추계종합학술발표회 우수논문상 수상

---

# 📚 What I Learned

* YOLOv8과 U-Net을 결합한 실시간 AI 시스템 설계 경험
* Fine-Tuning 기반 객체 탐지 성능 개선 경험
* 세그멘테이션과 객체 탐지 융합 처리 경험
* Agile 및 V-모델 기반 협업 및 프로젝트 관리 경험
* 테스트 기반 품질 개선 프로세스 경험

---

# 📄 Document

* [세종 테크노 파크 알고리즘 보고서](세종%20테크노%20파크%20알고리즘%20보고서.pdf)
