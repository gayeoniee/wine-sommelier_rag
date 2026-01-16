import streamlit as st
from wine_logic import get_wine_rag_chain

# 1. 페이지 설정: 'centered'로 설정하여 시선을 중앙으로 모읍니다.
st.set_page_config(
    page_title="Wine Sommelier",
    page_icon="🍷",
    layout="centered" 
)

# 2. 고밀도 커스텀 CSS (이 부분이 디자인의 90%를 결정합니다)
st.markdown("""
    <style>
    /* 1. 배경 그래디언트 - 와인의 깊은 풍미를 담은 색상 */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(180deg, #FDFBFB 0%, #EBEDEE 100%);
    }
    
    /* 2. 메인 컨테이너 - 종이가 떠 있는 듯한 카드 효과 */
    .main .block-container {
        max-width: 750px;
        background-color: white;
        padding: 3rem;
        border-radius: 25px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.05);
        margin-top: 40px;
        margin-bottom: 40px;
    }

    /* 3. 제목 디자인 - 굵고 신뢰감 있는 폰트 */
    h1 {
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        color: #1A1A1A;
        letter-spacing: -2px;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    /* 4. 필터 상태 바 - 칩(Chip) 스타일 */
    .filter-info {
        background-color: #F0F2F6;
        padding: 8px 15px;
        border-radius: 50px;
        font-size: 0.85rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }

    /* 5. 채팅 버블 커스텀 - 둥글고 부드러운 느낌 */
    [data-testid="stChatMessage"] {
        border-radius: 18px !important;
        padding: 15px;
        margin-bottom: 12px;
        border: 1px solid #F0F2F6;
    }
    </style>
    """, unsafe_allow_html=True)

# 3. 사이드바 (깔끔하게 정돈)
with st.sidebar:
    st.title("🍷 Sommelier Panel")
    st.write("전문적인 필터링으로 최적의 와인을 찾습니다.")
    st.markdown("---")
    price_range = st.slider("Budget ($)", 0, 500, (30, 100))
    min_points = st.select_slider("Rating", options=list(range(80, 101)), value=90)
    st.markdown("---")
    if st.button("🔄 Reset Session", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 4. 메인 헤더
st.title("Wine Sommelier")
# HTML 칩 스타일로 필터 정보 표시
st.markdown(f'<div class="filter-info">📍 현재 필터: {price_range[0]} - {price_range[1]} USD | {min_points}점 이상</div>', unsafe_allow_html=True)

# 지식 베이스 및 메시지 초기화
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = get_wine_rag_chain()

if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "반갑습니다. 당신의 취향을 분석하여 최고의 와인을 제안하는 소믈리에입니다. 무엇을 도와드릴까요?"}]

# 대화 기록 렌더링
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 5. 하단 채팅 입력창
if prompt := st.chat_input("프랑스 말고 50달러 이하 가성비 좋은 레드 와인 추천해줘"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("🔍 분석 엔진 가동 중...", expanded=False) as status:
            # 와인 관련 질문인지 검증
            from langchain_openai import ChatOpenAI
            validation_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
            validation_prompt = f"""다음 질문이 와인 추천/선택/정보와 관련된 질문인지 판단하세요.
와인 관련 질문이면 "YES", 아니면 "NO"만 답변하세요.

질문: {prompt}

답변 (YES 또는 NO만):"""

            validation_result = validation_llm.invoke(validation_prompt).content.strip().upper()

            if "NO" in validation_result:
                response = "죄송합니다. 저는 와인 추천 전문 소믈리에입니다. 와인 선택과 추천에 관한 질문만 답변드릴 수 있습니다. 🍷\n\n어떤 와인을 찾으시나요? 예를 들어:\n- 가성비 좋은 레드 와인 추천해줘\n- 해산물에 어울리는 화이트 와인\n- 50달러 이하 프랑스 와인"
                status.update(label="와인 질문이 아닙니다", state="error")
            else:
                # 지역 키워드 전처리
                region_mapping = {
                    "유럽": "European countries (France, Italy, Spain, Portugal, Germany, Austria, Greece)",
                    "미국": "United States (US only)",
                    "남미": "South American countries (Chile, Argentina)",
                    "오세아니아": "Oceania (Australia, New Zealand)",
                }

                enhanced_prompt = prompt
                for kr_region, en_region in region_mapping.items():
                    if kr_region in prompt:
                        enhanced_prompt = enhanced_prompt.replace(kr_region, en_region)

                enhanced_prompt = f"{enhanced_prompt} (예산: ${price_range[0]}-${price_range[1]}, 평점: {min_points}점 이상)"
                response = st.session_state.rag_chain.invoke(enhanced_prompt)
                status.update(label="분석 완료", state="complete")

        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})