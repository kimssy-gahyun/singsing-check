
import streamlit as st
import numpy as np

from PIL import Image, ImageOps
from tf_keras.models import load_model


# 페이지 설정
st.set_page_config(
    page_title="싱싱체크",
    page_icon="🍏",
    layout="centered"
)


# 모델 로딩 (앱 실행 중 재사용)
@st.cache_resource
def load_ai_model():
    return load_model("model/keras_Model.h5", compile=False)


# 클래스 정보 로딩
@st.cache_data
def load_labels():
    with open("model/labels.txt", "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


# 이미지 전처리 및 추론
def predict(image, model, labels):
    image = image.convert("RGB")

    # Teachable Machine 입력 크기
    image = ImageOps.fit(
        image,
        (224, 224),
        Image.Resampling.LANCZOS
    )

    image_array = np.asarray(image).astype(np.float32)

    # Teachable Machine 정규화
    normalized_image = (image_array / 127.5) - 1

    data = np.expand_dims(normalized_image, axis=0)

    prediction = model.predict(data, verbose=0)[0]

    results = {}

    for i, label in enumerate(labels):
        # "0 Fresh" -> "Fresh"
        class_name = label.split(" ", 1)[-1]
        results[class_name] = float(prediction[i])

    return results


# UI
st.title("🍏 싱싱체크")
st.write("AI 이미지 분석으로 과일·채소의 신선·부패 상태를 판별합니다.")

st.info("사진을 업로드하거나 카메라로 식재료를 촬영해 주세요.")

model = load_ai_model()
labels = load_labels()

input_method = st.radio(
    "이미지 입력 방식",
    ["사진 업로드", "카메라 촬영"],
    horizontal=True
)

if input_method == "사진 업로드":
    image_file = st.file_uploader(
        "식재료 사진을 선택해 주세요.",
        type=["jpg", "jpeg", "png"]
    )
else:
    image_file = st.camera_input("식재료를 촬영해 주세요.")

if image_file is not None:
    image = Image.open(image_file)

    st.image(
        image,
        caption="입력 이미지",
        use_container_width=True
    )

    if st.button("🔍 신선도 판별하기", type="primary", use_container_width=True):

        with st.spinner("AI가 이미지를 분석하고 있어요..."):
            results = predict(image, model, labels)

        predicted_class = max(results, key=results.get)
        confidence = results[predicted_class]

        st.divider()

        if predicted_class.lower() == "fresh":
            st.success("🍏 신선한 상태로 판별되었어요!")
        else:
            st.error("🍂 부패 상태로 판별되었어요.")

        st.metric(
            "예측 신뢰도",
            f"{confidence * 100:.2f}%"
        )

        st.subheader("클래스별 예측 확률")

        for class_name, probability in results.items():
            st.write(f"{class_name}: {probability * 100:.2f}%")
            st.progress(probability)

        st.caption(
            "본 결과는 이미지 기반 AI 판별이며, "
            "실제 식품의 섭취 안전성을 보장하지 않습니다."
        )
