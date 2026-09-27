
import streamlit as st
import numpy as np

from io import BytesIO

from PIL import Image, ImageOps
from tf_keras.models import load_model


# 페이지 설정
st.set_page_config(
    page_title="SINGSING CHECK",
    page_icon="🍏",
    layout="centered"
)


# 모델 로딩 (앱 실행 중 재사용)
@st.cache_resource
def load_ai_model():
    return load_model("model/keras_model.h5", compile=False)


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


# --------------------------------------

# 화면 스타일 설정
# 화이트 기반 화면에 입력 카드와 결과 카드를 구분하고 모바일 여백을 조정
# HTML과 CSS는 화면 표현에만 사용하며 모델 입력과 출력은 변경하지 않음
st.markdown(
    """
    <style>
    /* Streamlit 기본 헤더 숨김 */
    [data-testid="stHeader"] {
        display: none !important;
    }

    /* 기본 상단 장식 제거 */
    [data-testid="stDecoration"] {
        display: none !important;
    }

    /* 기본 폰트 */
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

    .stApp {
        font-family: 'Pretendard', sans-serif;
    }

    /* 제목 옆 자동 앵커 링크 숨기기 */
    [data-testid="stHeaderActionElements"] {
        display: none !important;
    }

    /* 기본 화면 : 화이트와 중립 회색을 사용하고 그린은 포인트로만 적용 */
    .stApp, [data-testid="stHeader"] {
        background-color: #FAFAFA;
        color: #363D38;
        color-scheme: light;
    }
    [data-testid="stMainBlockContainer"] {
        max-width: 720px;
        padding: 3rem 1.5rem;
    }
    .stApp h1, .stApp h2, .stApp h3,
    .stApp [data-testid="stWidgetLabel"] {
        color: #363D38;
    }
    .stApp [data-testid="stCaptionContainer"] {
        color: #626963;
    }
    .service-name {
        margin: 0 0 0.6rem;
        color: #438653;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    .stApp h1 {
        font-size: clamp(1.7rem, 5vw, 2.3rem);
        line-height: 1.35;
        letter-spacing: -0.045em;
        word-break: keep-all;
        overflow-wrap: break-word;
    }
    .hero-title {
        color: #32865B;
    }
    .stApp h3 {
        font-size: 1.15rem;
        line-height: 1.5;
    }
    /* 입력 카드는 화이트 배경으로 구분 */
    .st-key-image_input {
        background-color: #FFFFFF;
        border: 1px solid #E3E6E3;
        border-radius: 12px;
        padding: 1.25rem;
    }
    [data-testid="stFileUploaderDropzone"] {
        background-color: #FFFFFF;
        color: #363D38;
        border-radius: 8px;
    }
    .stApp button[kind="secondary"] {
        background-color: #FFFFFF;
        color: #363D38;
        border-color: #D7DCD7;
    }
    .stApp [data-testid="stExpander"] details {
        background-color: #FFFFFF;
        color: #363D38;
        border-color: #E3E6E3;
    }
    /* 모바일에서 누르기 편한 버튼 높이와 키보드 포커스 표시 */
    .stButton > button[kind="primary"] {
        min-height: 3.1rem;
        background-color: #438653;
        color: #FFFFFF;
        border: 1px solid #438653;
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #356B42;
        border-color: #356B42;
        color: #FFFFFF;
    }
    .stButton > button[kind="primary"]:focus-visible {
        outline: 3px solid #438653;
        outline-offset: 3px;
    }
    /* 결과 카드 : 배경은 화이트로 유지하고 상태와 확률에만 색상 적용 */
    .freshness-result {
        margin-top: 0.75rem;
        padding: 1.5rem;
        background-color: #FFFFFF;
        border: 1px solid #E3E6E3;
        border-radius: 12px;
        color: #363D38;
    }
    .freshness-result.fresh {
        --result-color: #356B42;
        --result-tint: #EDF5EE;
    }
    .freshness-result.stale {
        --result-color: #934E38;
        --result-tint: #FAEFE9;
    }
    .freshness-result .result-status {
        display: inline-block;
        padding: 0.3rem 0.65rem;
        border-radius: 6px;
        background-color: var(--result-tint);
        color: var(--result-color);
        font-size: 0.9rem;
        font-weight: 600;
    }
    .freshness-result h2 {
        margin: 0.85rem 0 1.2rem;
        padding: 0;
        font-size: clamp(1.25rem, 4vw, 1.5rem);
        line-height: 1.45;
        color: #363D38;
        word-break: keep-all;
    }
    .freshness-result .result-confidence {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 0.25rem 1rem;
        margin: 0;
        color: #626963;
        font-size: 0.9rem;
    }
    .result-confidence strong {
        color: var(--result-color);
        font-size: clamp(1.8rem, 7vw, 2.2rem);
        line-height: 1.2;
        font-variant-numeric: tabular-nums;
    }
    /* 작은 화면에서는 좌우 여백과 카드 안쪽 여백을 줄임 */
    @media (max-width: 480px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1.75rem 1rem 2rem;
        }
        .st-key-image_input, .freshness-result {
            padding: 1rem;
        }
        .service-name {
            font-size: 1.05rem;
        }
        [data-testid="stFileUploaderDropzone"] {
            flex-wrap: wrap;
            gap: 0.75rem;
        }
    }
    /* 브랜드는 아이콘, 한글 이름, 영문 이름 순서로 배치 */
    .brand-header { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1.75rem; }
    .brand-icon { font-size: 1.8rem; }
    .brand-name { font-size: 1.25rem; font-weight: 750; color: #252A27; }
    .brand-subtitle, .result-eyebrow { font-size: 0.72rem; letter-spacing: 0.14em; color: #626963; }
    /* 결과 섹션 소제목 */
    .stApp p.result-eyebrow {
        margin: 12px 0 8px !important;
        padding: 0 !important;
        color: #69766F !important;
        font-size: 0.85rem !important;
        font-weight: 600;
        letter-spacing: 0.15em;
        line-height: 1.4;
    }
    .stApp .main-copy { margin: 0; padding: 0 0 0.75rem; }
    .main-copy span { color: #438653; }
    /* 복사본 미리보기만 제한 : 모델에 전달하는 원본에는 영향을 주지 않음 */
    .st-key-image_preview { background: #F7F8F7; border-radius: 8px; padding: 0; overflow: hidden; position: relative; gap: 0; }
    .st-key-image_preview [data-testid="stImage"] { width: 100%; margin: 0; }
    /* 높이는 320px로 유지하고 사진 전체가 보이도록 비율을 맞춰 축소 */
    .st-key-image_preview img {
        display: block;
        width: 100%;
        height: 320px;
        max-height: 320px;
        object-fit: contain;
        object-position: center;
        border-radius: 8px;
    }
    .result-icon { font-size: 1.6rem; margin-right: 0.4rem; color: var(--result-color); }
    .result-track { height: 6px; margin-top: 1rem; background: #EDF0ED; border-radius: 3px; overflow: hidden; }
    .result-fill { height: 100%; background: var(--result-color); border-radius: inherit; }
    .freshness-result .result-confidence strong { font-size: clamp(2.2rem, 9vw, 3rem); }
    @media (max-width: 480px) {
        .brand-header { margin-bottom: 1.4rem; }
        .brand-name { font-size: 1.15rem; }
        .freshness-result .result-confidence { align-items: flex-start; flex-direction: column; gap: 0.5rem; }
    }
    /* 결과 카드와 상세 예측 확률 사이 간격 */
    .stApp .freshness-result {
        margin-bottom: 12px !important;
    }
    /* 사진 등록 제목과 단계 배지 : 입력 영역에만 적용 */
    .photo-section-heading { display: flex; align-items: center; justify-content: space-between; gap: 1rem; }
    .photo-section-heading h3 { margin: 0; padding: 0; font-size: 1.15rem; font-weight: 700; }
    .photo-step { padding: 0.3rem 0.6rem; border-radius: 6px; background: #F1F5F2; color: #438653; font-size: 0.7rem; font-weight: 700; letter-spacing: 0.08em; }
    /* 작은 모바일 화면에서도 카드 두 개를 같은 너비로 나란히 유지 */
    .st-key-input_choices [data-testid="stHorizontalBlock"] { flex-wrap: nowrap !important; gap: 0.75rem; }
    .st-key-input_choices [data-testid="stColumn"] { flex: 1 1 0 !important; width: 0 !important; min-width: 0 !important; }
    /* 실제 Streamlit 버튼 전체가 클릭 영역이며 아이콘도 글자색을 따름 */
    .st-key-input_choices .stButton > button {
        width: 100%; min-height: 120px; padding: 1rem 0.4rem;
        display: flex; flex-direction: column; justify-content: center; gap: 0.65rem;
        border: 1px solid #E0E5E2; border-radius: 12px;
        background: #FFFFFF; color: #68736D; box-shadow: none;
    }
    .st-key-input_choices .stButton > button p { font-size: 0.95rem; font-weight: 600; color: inherit; word-break: keep-all; }
    .st-key-input_choices .stButton > button [data-testid="stIconMaterial"] { font-size: 2rem; color: inherit; }
    .st-key-input_choices .stButton > button[kind="primary"],
    .st-key-input_choices .stButton > button[kind="primary"]:hover,
    .st-key-input_choices .stButton > button[kind="primary"]:active {
        background: #ECF8F1; border-color: #83B99A; color: #438653;
    }
    .st-key-input_choices .stButton > button[kind="secondary"]:hover {
        background: #FFFFFF; border-color: #AEBBB3; color: #68736D;
    }
    .st-key-input_choices .stButton > button:focus-visible { outline: 3px solid #438653; outline-offset: 3px; }
    @media (max-width: 480px) {
        .st-key-input_choices [data-testid="stHorizontalBlock"] { gap: 0.6rem; }
        .st-key-input_choices .stButton > button { min-height: 108px; }
        .st-key-input_choices .stButton > button p { font-size: 0.875rem; }
    }
    /* 사진 등록 전에는 클릭을 막고 회색으로 비활성 상태 표시 */
    .stButton > button[kind="primary"]:disabled,
    .stButton > button[kind="primary"]:disabled:hover,
    .stButton > button[kind="primary"]:disabled:active {
        background-color: #E8EBE9;
        border-color: #E8EBE9;
        color: #737C76;
        opacity: 1;
        cursor: not-allowed;
        box-shadow: none;
    }
    /* 파일명·용량 행만 숨김 : 입력 위젯과 파일 선택 버튼은 계속 표시 */
    .st-key-image_input [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
    .st-key-image_input [data-testid="stFileUploader"] [data-testid="stFileChip"] {
        display: none;
    }
    /* 업로드 후 파일 목록에 붙는 + 버튼만 숨김 */
    .st-key-image_input [data-testid="stFileUploader"] [data-testid="stFileChips"] button[aria-label="Add files"] {
        display: none;
    }
    /* 업로드된 파일이 있을 때만 빈 파일 영역의 높이와 주변 간격 축소 */
    .st-key-image_input:has([data-testid="stFileChip"]) {
        gap: 0.5rem;
    }
    .st-key-image_input [data-testid="stFileUploaderDropzone"]:has([data-testid="stFileChips"]) {
        min-height: 0;
        padding: 0;
        gap: 0;
        border: 0;
    }
    .st-key-image_input [data-testid="stFileChips"] {
        min-height: 0;
        margin: 0;
        gap: 0;
    }
    /* 미리보기의 버튼 요소만 absolute로 배치 : 입력 위젯과 이미지는 그대로 유지 */
    .st-key-image_preview > [data-testid="stElementContainer"]:has(.stButton) {
        position: absolute;
        top: 12px;
        right: 12px;
        width: 44px;
        z-index: 2;
        margin: 0;
    }
    .st-key-image_preview .stButton > button {
        width: 44px;
        height: 44px;
        min-height: 44px;
        padding: 0;
        border: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.94);
        color: #363D38;
    }
    .st-key-image_preview .stButton > button p { font-size: 1.7rem; line-height: 1; }
    .st-key-image_preview .stButton > button:hover { background: #FFFFFF; color: #252A27; }
    .st-key-image_preview .stButton > button:focus-visible { outline: 3px solid #438653; outline-offset: 2px; }

    /* 주의 문구 */
    .safety-notice {
        margin: 12px 0 24px;
        color: #8A938E;
        font-size: 0.8rem !important;
        line-height: 1.7;
        font-weight: 400;
        word-break: keep-all;
    }
    /* 사진 등록 제목과 안내 문구 사이 여백을 업로드 전후 동일하게 유지 */
    .st-key-image_input [data-testid="stCaptionContainer"] {
        margin-top: 0.85rem !important;
        margin-bottom: 0.85rem !important;
    }
    /* 결과 화면 사진의 Full Screen 컨트롤 숨김 */
    .st-key-analysis_image_preview [data-testid="stElementToolbar"] {
        display: none !important;
    }    /* 로고 전체가 하나의 홈 버튼이며, 시각적으로는 헤더처럼 표시 */
    .st-key-brand_home { margin-bottom: 1.75rem; }
    .st-key-brand_home .stButton > button {
        min-height: 0;
        width: auto;
        padding: 0;
        border: 0;
        background: transparent;
        color: #252A27;
        box-shadow: none;
        display: flex;
        flex-direction: column;
        align-items: flex-start;
        text-align: left;
    }
    .st-key-brand_home .stButton > button p {
        margin: 0;
        color: #252A27;
        font-size: 1.2rem;
        line-height: 1.2;
        font-weight: 750;
        white-space: nowrap;
    }
    .st-key-brand_home .stButton > button::after {
        content: "SINGSING CHECK";
        display: block;
        margin-top: 0.22rem;
        padding-left: 2.25rem;
        color: #626963;
        font-size: 0.625rem;
        line-height: 1;
        font-weight: 500;
        letter-spacing: 0.14em;
    }
    .st-key-brand_home .stButton > button:hover,
    .st-key-brand_home .stButton > button:hover p {
        background: transparent;
        color: #438653;
    }
    .st-key-brand_home .stButton > button:hover::after { color: #626963; }
    .st-key-brand_home .stButton > button:focus-visible {
        outline: 3px solid #438653;
        outline-offset: 4px;
    }
    @media (max-width: 480px) {
        .st-key-brand_home { margin-bottom: 1.4rem; }
        .st-key-brand_home .stButton > button p { font-size: 1.1rem; }
    }    /* 공통 하단 저작권 문구 */
    .app-footer {
        margin: 2rem 0 0;
        padding: 1rem 0 0.25rem;
        border-top: 1px solid #E3E6E3;
        color: #8A938E;
        font-size: 0.72rem;
        line-height: 1.4;
        text-align: center;
    }    </style>
    """,
    unsafe_allow_html=True # 위에서 작성한 CSS를 화면에 적용
)

# --------------------------------------

# 사진 삭제는 위젯 값을 직접 바꾸지 않고 새 key의 빈 입력 위젯으로 교체
# 업로드·카메라 위젯 자체는 계속 표시하여 바로 새 사진을 등록할 수 있음
def clear_photo():
    st.session_state.input_version = st.session_state.get("input_version", 0) + 1
    st.session_state.analysis_results = None


# 상단 브랜드 클릭 시 입력 초기 화면으로 이동
# 현재 결과와 사진 입력 key를 초기화하고 새 업로드 위젯을 생성

def go_home():
    clear_photo()
    st.session_state.pop("input_method", None)
    st.session_state.screen = "input"

# 화면 전환용 임시 상태 초기화
# 현재 화면과 판별 결과만 관리하고, 사진 입력은 Streamlit 기본 위젯에 맡김
if "screen" not in st.session_state:
    st.session_state.screen = "input"
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# 두 화면에서 공통으로 사용하는 브랜드 헤더
with st.container(key="brand_home"):
    st.button("🍏 싱싱체크", key="brand_home_button", on_click=go_home)
# --------------------------------------

# 화면 1 : 사진 등록 및 판별
if st.session_state.screen == "input":
    st.markdown(
        """
        <div class="hero">
            <h1>
                싱싱한지 궁금할 땐, <span class="hero-title">싱싱체크!</span>
            </h1>
            <p class="hero-description">
                AI가 사진 속 과일과 채소의 상태를 분석해 드려요.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 기존 모델과 클래스 정보 불러오기 : 캐시된 모델은 그대로 재사용
    model = load_ai_model()
    labels = load_labels()

    # 입력 위젯은 사진 선택 후에도 계속 표시하여 기본 삭제·재선택 기능 유지
    # 업로드와 카메라는 Streamlit 기본 위젯을 그대로 사용
    image = None
    with st.container(border=False, key="image_input"):
        st.markdown(
            '<div class="photo-section-heading"><h3>사진 등록</h3>'
            '</div>',
            unsafe_allow_html=True
        )
        st.caption("과일이나 채소가 잘 보이는 사진을 사용해 주세요.")

        # 기본 버튼을 카드로 표현하여 카드 전체를 클릭하거나 키보드로 선택 가능
        # 콜백은 화면을 다시 그리기 전에 실행되어 선택 색상과 입력 위젯을 함께 변경
        if "input_method" not in st.session_state:
            st.session_state.input_method = "사진 업로드"
        input_method = st.session_state.input_method

        # 이 컨테이너 안의 두 열에만 모바일 가로 배치 CSS 적용
        with st.container(key="input_choices"):
            upload_column, camera_column = st.columns(2, gap="small")
            with upload_column:
                st.button(
                    "사진 업로드",
                    icon=":material/add_photo_alternate:",
                    key="choose_upload",
                    type="primary" if input_method == "사진 업로드" else "secondary",
                    use_container_width=True,
                    on_click=st.session_state.update,
                    kwargs={"input_method": "사진 업로드"}
                )
            with camera_column:
                st.button(
                    "카메라 촬영",
                    icon=":material/photo_camera:",
                    key="choose_camera",
                    type="primary" if input_method == "카메라 촬영" else "secondary",
                    use_container_width=True,
                    on_click=st.session_state.update,
                    kwargs={"input_method": "카메라 촬영"}
                )

        input_version = st.session_state.get("input_version", 0)
        # 선택한 방식의 기본 입력 위젯을 항상 표시
        # 파일 선택 버튼과 카메라 재촬영 기능은 유지하고 파일명 행만 CSS로 정리
        if input_method == "사진 업로드":
            image_file = st.file_uploader(
                "식재료 사진",
                type=["jpg", "jpeg", "png"],
                key=f"uploaded_image_{input_version}",
                label_visibility="collapsed" # 안내 문구와 빈 라벨 공간 제거
            )
        else:
            image_file = st.camera_input(
                "식재료를 촬영해 주세요.",
                key=f"camera_image_{input_version}"
            )

        # 기본 위젯에서 삭제하면 None이 반환되어 미리보기와 판별 버튼도 사라짐
        if image_file is not None:
            try:
                image = Image.open(image_file)
                # 원본은 추론용으로 유지하고, 미리보기용 복사본만 축소
                preview_image = image.copy()
                preview_image.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
                with st.container(key="image_preview"):
                    st.image(preview_image, use_container_width=True)
                    st.button(
                        "×",
                        key="remove_photo",
                        help="사진 삭제",
                        on_click=clear_photo
                    )
            except (OSError, ValueError):
                image = None
                st.error("사진을 읽을 수 없어요. JPG 또는 PNG 사진을 다시 선택해 주세요.")

    # 사진이 없거나 삭제된 경우에는 판별 버튼을 표시하지 않음
    if image is not None and st.button(
        "AI 신선도 확인하기",
        type="primary",
        use_container_width=True
    ):
        with st.spinner("사진을 확인하고 있어요..."):
            # preview_image가 아닌 원본 image를 기존 predict 함수에 전달
            results = predict(image, model, labels)

        # 결과 화면에 판별 결과만 전달하고, 홈 복귀 시 기본 입력 위젯을 초기화
        st.session_state.analysis_image_bytes = image_file.getvalue() # 결과 화면용 임시 이미지
        st.session_state.analysis_results = results
        st.session_state.screen = "result"
        st.rerun()

# --------------------------------------

# 화면 2 : AI 판별 결과
else:
    results = st.session_state.analysis_results

    # 임시 결과가 없는 경우 입력 화면으로 복귀
    if not results:
        st.session_state.screen = "input"
        st.rerun()

    # 판별에 사용한 사진을 결과 화면에서도 미리보기로 표시
    analysis_image_bytes = st.session_state.get("analysis_image_bytes")
    if analysis_image_bytes is not None:
        with st.container(key="analysis_image_preview"):
            analysis_image = Image.open(BytesIO(analysis_image_bytes))
            st.image(analysis_image, use_container_width=True)

    st.markdown('<p class="result-eyebrow">AI ANALYSIS RESULT</p>', unsafe_allow_html=True)

    # 기존 클래스 선택 로직 유지 : 가장 높은 확률의 클래스가 최종 결과
    predicted_class = max(results, key=results.get)
    confidence = results[predicted_class]

    if predicted_class.lower() == "fresh":
        result_style = "fresh"
        result_label = "Fresh"
        result_title = "신선한 상태로 보여요"
        result_icon = "✓"
    else:
        result_style = "stale"
        result_label = "Stale"
        result_title = "부패한 상태로 보여요"
        result_icon = "!"

    # 색상 외에도 아이콘과 상태 문구로 구분
    # HTML에는 정해진 문구와 숫자만 삽입하며 진행 막대는 예측 확률을 표현
    st.markdown(
        f"""
        <section class="freshness-result {result_style}" aria-label="신선도 판별 결과">
            <span class="result-icon" aria-hidden="true">{result_icon}</span>
            <span class="result-status">{result_label}</span>
            <h2>{result_title}</h2>
            <p class="result-confidence"><span>모델 예측 확률</span><strong>{confidence * 100:.2f}%</strong></p>
            <div class="result-track" aria-hidden="true">
                <div class="result-fill" style="width: {confidence * 100:.2f}%"></div>
            </div>
        </section>
        """,
        unsafe_allow_html=True
    )

    with st.expander("상세 예측 확률 보기"):
        for class_name, probability in results.items():
            st.write(f"{class_name}: {probability * 100:.2f}%")
            st.progress(probability)

    st.markdown(
        """
        <p class="safety-notice">
            본 결과는 이미지 기반 AI 판별입니다.<br>
            예측 확률은 모델의 분류 확률이며, 부패 정도나 실제 섭취 안전성을 의미하지 않습니다.
        </p>
        """,
        unsafe_allow_html=True
    )

    if st.button(
        "다른 식재료 확인하기",
        use_container_width=True,
        type="primary",
        key="return_to_input"
    ):
        # 실제 입력 위젯 key를 갱신해 이전 사진을 새 입력에 재사용하지 않음
        st.session_state.input_version = st.session_state.get("input_version", 0) + 1
        st.session_state.analysis_results = None
        st.session_state.pop("analysis_image_bytes", None)
        st.session_state.screen = "input"
        st.rerun()
# 모든 화면에서 공통으로 표시하는 하단 저작권 문구
st.markdown(
    '<footer class="app-footer">© 2026 SINGSING CHECK · Developed by Gahyun Kim</footer>',
    unsafe_allow_html=True
)