# 🍏 싱싱체크 (SINGSING CHECK)

**딥러닝 이미지 분석을 활용한 과일·채소 신선도 판별 웹 서비스**

싱싱체크는 과일이나 채소의 사진을 업로드하면 AI 모델이 이미지의 특징을 분석하여 **Fresh(신선)** 또는 **Stale(부패)** 상태로 분류하는 웹 애플리케이션입니다.

Google Teachable Machine으로 이미지 분류 모델을 학습하고, Streamlit을 활용하여 별도의 설치 없이 웹 브라우저에서 사용할 수 있도록 구현했습니다.

### 🔗 서비스 바로가기

**[싱싱체크 이용하기](https://singsing-check.streamlit.app/)**

## 1. 주요 기능

- **사진 업로드:** JPG, JPEG, PNG 이미지 등록
- **카메라 촬영:** 기기 카메라로 식재료 사진 촬영
- **AI 신선도 판별:** 업로드한 이미지를 Fresh / Stale 두 클래스로 분류
- **결과 시각화:** 판별 결과, 모델 예측 확률 및 상세 클래스별 확률 표시
- **반응형 UI:** PC와 모바일 환경에서 이용 가능한 화면 구성

## 2. 사용 방법

1. 웹사이트에 접속합니다.
2. `사진 업로드` 또는 `카메라 촬영`을 선택합니다.
3. 과일이나 채소가 잘 보이는 사진을 등록합니다.
4. `AI 신선도 확인하기` 버튼을 누릅니다.
5. 판별 결과와 모델 예측 확률을 확인합니다.
6. `다른 식재료 확인하기`를 눌러 새로운 사진을 분석할 수 있습니다.

## 3. 딥러닝 모델

### 데이터셋

Kaggle의 **Fresh and Stale Images of Fruits and Vegetables** 데이터셋을 활용했습니다.

- 전체 이미지: 11,986장
- 분류 클래스: Fresh / Stale

### 모델 학습

Google Teachable Machine을 사용하여 이미지 분류 모델을 학습했습니다.

| 항목 | 내용 |
|---|---|
| 학습 도구 | Google Teachable Machine |
| 분류 방식 | 이미지 이진 분류 |
| 클래스 | Fresh, Stale |
| 선택 모델 | Epoch 20 |
| 테스트 정확도 | 98.11% |

### 혼동 행렬 (Confusion Matrix)

| 실제 \ 예측 | Fresh | Stale |
|---|---:|---:|
| Fresh | 908 | 9 |
| Stale | 25 | 857 |

※ 위 성능은 프로젝트에서 사용한 테스트 데이터 기준이며, 실제 사용 환경의 모든 이미지에 동일한 정확도를 보장하지 않습니다.

## 4. 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python |
| Web Framework | Streamlit |
| AI Model | Google Teachable Machine |
| Deep Learning | TensorFlow / Keras |
| Image Processing | Pillow, NumPy |
| Deployment | Streamlit Community Cloud |
| Version Control | Git, GitHub |

## 5. 이미지 분석 과정

```text
이미지 업로드 또는 카메라 촬영
           ↓
       RGB 변환
           ↓
     224 × 224 리사이즈
           ↓
      이미지 정규화
           ↓
       AI 모델 추론
           ↓
   Fresh / Stale 분류
           ↓
   결과 및 예측 확률 표시
```

모델 입력 이미지는 Teachable Machine의 입력 형식에 맞게 전처리하며, 모델이 출력한 클래스별 확률 중 가장 높은 값을 최종 분류 결과로 사용합니다.

## 6. 로컬 실행 방법

### 저장소 복제

```bash
git clone https://github.com/kimssy-gahyun/singsing-check.git
cd singsing-check
```

### 의존성 설치

```bash
pip install -r requirements.txt
```

### 애플리케이션 실행

```bash
streamlit run app.py
```

실행 후 터미널에 표시되는 로컬 주소로 접속하면 됩니다.

모델 파일과 클래스 정보 파일은 각각 다음 경로에 있어야 합니다.

```text
model/keras_model.h5
model/labels.txt
```

## 7. 프로젝트 구조

```text
singsing-check/
├── app.py
├── requirements.txt
├── model/
│   ├── keras_model.h5
│   └── labels.txt
└── README.md
```

## 8. 이용 시 유의사항

싱싱체크는 **이미지 기반 딥러닝 분류 프로젝트**입니다.

화면에 표시되는 예측 확률은 모델이 해당 클래스로 분류한 확률을 의미하며, 부패 정도나 실제 섭취 안전성을 나타내지 않습니다. 촬영 환경, 조명, 이미지 품질 및 학습 데이터와의 차이에 따라 판별 결과가 달라질 수 있습니다.

실제 식재료의 섭취 여부는 AI 판별 결과만으로 결정하지 마세요.

---

**싱싱체크 | SINGSING CHECK**  
AI를 활용해 일상 속 식재료의 상태를 간편하게 확인하는 이미지 분류 프로젝트입니다.