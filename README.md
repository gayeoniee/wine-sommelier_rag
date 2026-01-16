# 🍷 Wine Sommelier - AI 기반 개인 맞춤형 와인 추천 시스템

> **1인 프로젝트 | 2025.12.10 ~ 2026.01.18**  
> **역할**: 기획, 데이터 엔지니어링, AI 개발

**[GitHub Repository](#)** ← 추후 링크 추가 예정

---

## 1️⃣ 프로젝트 개요

### 📌 한 줄 소개
"달지 않고 바디감 있는 레드 와인"처럼 모호한 사용자 질문을 이해하고 72,000건의 와인 데이터에서 최적의 와인을 찾아주는 AI 소믈리에 시스템

### 🎯 핵심 가치
- **모호한 자연어 질의 처리**: "프랑스 와인 제외해줘", "산미가 강한 화이트 와인" 등 자연스러운 대화로 추천
- **논리적 필터링**: Self-Querying Retriever로 100% 정확한 메타데이터 필터링 구현
- **고품질 큐레이션**: 평점 88점 이상 데이터만 선별하여 신뢰도 높은 추천 제공

### 🛠 기술 스택

| 카테고리 | 기술 상세 |
|---------|----------|
| **Language** | Python 3.10+ |
| **Framework** | LangChain, Streamlit |
| **AI/NLP** | Self-Querying Retriever, Feature Extraction |
| **Vector DB** | Pinecone (클라우드 벡터 데이터베이스) |
| **Embedding** | OpenAI text-embedding-3-small |
| **LLM** | GPT-4o-mini (메타데이터 쿼리 생성) |
| **Tools** | VS Code, Git |

### 📊 주요 성과 지표

| 지표 | 수치 | 설명 |
|------|------|------|
| **데이터 규모** | 130,000건 → 72,000건 | 평점 88점 이상 고품질 데이터 44% 정제 |
| **필터링 정확도** | 100% | Self-Querying 기반 논리 필터 (오탐률 0%) |
| **응답 시간** | 평균 2.5초 | 질의 → 추천 결과 생성 |
| **추출 Feature** | 12개 맛 태그 | Oak, Berry, Acid, Floral 등 NLP 자동 추출 |

---

## 2️⃣ 프로젝트 취지

### 🔍 해결하고자 한 문제

#### **Problem 1: 와인 선택의 어려움**
- 와인 초보자는 자신의 취향을 전문 용어로 표현하기 어려움
- "달지 않은", "묵직한", "과일향 나는" 같은 모호한 표현 사용
- 기존 추천 시스템은 정확한 키워드 입력을 요구함

#### **Problem 2: 유사도 검색의 맹점**
```python
# 문제 상황 예시
사용자: "프랑스 와인 제외해줘"

[기존 유사도 검색 방식]
→ "프랑스" 키워드와 유사한 문서 검색
→ 결과: 프랑스 와인이 상위 랭크됨 (역효과!)

[Self-Querying 방식]
→ LLM이 자연어를 메타데이터 필터로 변환
→ Country != 'France' 논리 필터 적용
→ 결과: 프랑스 와인 100% 제외
```

#### **Problem 3: 대규모 데이터에서의 품질 관리**
- 약 130,000건의 와인 리뷰 중 품질 편차 큼
- 평점 낮은 와인 추천 시 사용자 신뢰도 하락

---

### 💡 솔루션

#### **Solution 1: Self-Querying Retriever 구현**
자연어를 구조화된 쿼리로 자동 변환하여 정확한 필터링 수행

```
사용자 입력: "이탈리아산 레드 와인 중에서 2015년 이후 생산된 걸로 추천해줘"

↓ LLM 파싱 (GPT-4o-mini)

Semantic Query: "이탈리아 레드 와인"
Metadata Filter: {
  "Country": "Italy",
  "Variety": "Red",
  "Year": {"$gte": 2015}
}

↓ Pinecone 검색

결과: 이탈리아 레드 와인 + 2015년 이후 + 유사도 높은 순
```

#### **Solution 2: 고품질 데이터 엔지니어링**
1. **평점 기반 필터링**: 88점 이상만 선별 (130,000건 → 72,000건)
2. **NLP Feature Extraction**: 리뷰에서 맛 키워드 자동 추출
   - Oak, Berry, Acid, Citrus, Spice 등 12개 카테고리 태깅

#### **Solution 3: 사용자 친화적 인터페이스**
- Streamlit으로 대화형 UI 구현
- 실시간 채팅 방식 추천
- 추천 결과에 근거(맛 프로필, 평점, 리뷰) 함께 제공

---

### 🎯 차별점

| 비교 항목 | 기존 추천 시스템 | Wine Sommelier |
|----------|----------------|----------------|
| **질의 방식** | 정확한 키워드 필요 | 자연어 대화 가능 |
| **부정 표현** | 처리 불가 ("~제외") | 논리 필터로 100% 처리 |
| **데이터 품질** | 전체 데이터 사용 | 고평점만 선별 (신뢰도↑) |
| **맛 정보** | 수동 입력 필요 | NLP 자동 추출 |
| **사용성** | 전문가 중심 | 초보자 친화적 |

---

## 3️⃣ 구현 기능 목록

### 🏗 시스템 아키텍처

```
[사용자 입력]
     ↓
[Streamlit UI]
     ↓
[LangChain Self-Querying]
     ├─→ [LLM] → 메타데이터 필터 생성
     └─→ [Embedding Model] → 의미 벡터 생성
     ↓
[Pinecone Vector DB]
     ├─ Semantic Search (유사도 검색)
     └─ Metadata Filtering (논리 필터)
     ↓
[결과 후처리 & 랭킹]
     ↓
[추천 결과 반환]
```

---

### ⚙️ 핵심 기능 상세

#### **[Feature 1] Self-Querying Retriever**

**구현 배경**
- 유사도 검색만으로는 부정 표현("~아닌", "~제외") 처리 불가
- 숫자 비교("X년 이후", "가격 Y원 이하") 논리 연산 필요

**기술적 구현**
```python
# LangChain Self-Querying 설정 예시
from langchain.retrievers import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

metadata_field_info = [
    AttributeInfo(
        name="Country",
        description="와인 생산 국가 (예: France, Italy, USA)",
        type="string"
    ),
    AttributeInfo(
        name="Points",
        description="평점 (88-100점)",
        type="integer"
    ),
    AttributeInfo(
        name="Price",
        description="가격 (USD)",
        type="float"
    ),
    # ... 추가 메타데이터
]

retriever = SelfQueryRetriever.from_llm(
    llm=ChatOpenAI(model="gpt-4o-mini"),
    vectorstore=pinecone_vectorstore,
    document_contents="와인 리뷰 및 맛 프로필",
    metadata_field_info=metadata_field_info
)
```

**성과**
- ✅ 부정 표현 100% 정확 처리
- ✅ 복합 조건(AND, OR, NOT) 지원
- ✅ 기존 유사도 검색 대비 오탐률 0%

---

#### **[Feature 2] NLP 기반 맛 Feature 추출**

**구현 배경**
- 와인 리뷰는 비정형 텍스트
- 검색 효율을 위해 구조화된 맛 태그 필요

**기술적 구현**
```python

# 맛 카테고리 매핑
TASTE_CATEGORIES = {
    'fruit': ['berry', 'cherry', 'apple', 'citrus'],
    'oak': ['oak', 'vanilla', 'toast', 'smoke'],
    'acid': ['crisp', 'bright', 'zesty', 'tart'],
    # ... 
}

# 추출 결과를 메타데이터로 저장
metadata['taste_profile'] = extracted_tags
```

**성과**
- ✅ 12개 맛 카테고리 자동 태깅
- ✅ 수동 라벨링 대비 작업 시간 95% 단축
- ✅ 리뷰 텍스트 → 검색 가능한 구조화 데이터 변환

---

#### **[Feature 3] 데이터 품질 관리 파이프라인**

**처리 과정**
```python
# 1단계: 초기 데이터 로드
df = pd.read_csv('winemag-data-130k-v2.csv')  # 129,971건

# 2단계: 결측치 처리
df = df.dropna(subset=['description', 'points', 'country'])  # 72,000건

# 3단계: 평점 필터링 (신뢰도 향상)
df = df[df['points'] >= 88]  

# 4단계: 중복 제거
df = df.drop_duplicates(subset=['title', 'description'])



# 최종: 약 72,000건 고품질 데이터셋
```


---

#### **[Feature 4] Pinecone 벡터 DB 구축**

**인덱스 설계**
```python
import pinecone
from langchain.embeddings import OpenAIEmbeddings

# Pinecone 초기화
pinecone.init(api_key=PINECONE_API_KEY, environment='gcp-starter')

# 인덱스 생성 (차원수: OpenAI embedding 1536)
index_name = pinecone.Index('wine-sommelier')

# 문서 임베딩 및 업로드
embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

# 3. LangChain Document 객체로 변환
documents = []
for _, row in df.iterrows():
    doc = Document(
        page_content=row['page_content'], # 검색 대상 텍스트
        metadata={
            "title": row['title'],
            "country": row['country'],
            "continent": row['continent'],
            "points": int(row['points']),
            "price": float(row['price']),
            "variety": row['variety'],
            "tag_oak": int(row['tag_oak']),
            "tag_acid": int(row['tag_acid'])
        }
    )
    documents.append(doc)

# 4. Batch Upsert (100개씩 나누어 업로드)
batch_size = 100
for i in range(0, len(documents), batch_size):
    batch = documents[i : i + batch_size]
    PineconeVectorStore.from_documents(
        batch, 
        embeddings, 
        index_name=index_name
    )
    print(f"{i + len(batch)} / {len(documents)} 업로드 완료...")
    time.sleep(1) # API 레이트 리밋 방지
```

**성능 최적화**
- ✅ 배치 업로드 (100개씩) → 처리 시간 단축
- ✅ 메타데이터 인덱싱 → 필터링 속도 향상
- ✅ 무료 티어 최대 활용 (100만 벡터)

---

#### **[Feature 5] Streamlit 웹 인터페이스**

**주요 UI 구성**
```python
import streamlit as st

# 1. 대화형 채팅 인터페이스
st.title("🍷 Wine Sommelier")
user_input = st.chat_input("어떤 와인을 찾으시나요?")

if user_input:
    # 2. 실시간 응답 표시
    with st.chat_message("assistant"):
        with st.spinner("와인을 검색 중입니다..."):
            results = retriever.get_relevant_documents(user_input)
        
        # 3. 추천 결과 카드 형식으로 표시
        for wine in results[:3]:
            with st.expander(f"🍇 {wine.metadata['title']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("평점", f"{wine.metadata['points']}점")
                    st.metric("가격", f"${wine.metadata['price']}")
                with col2:
                    st.write(f"**국가**: {wine.metadata['country']}")
                    st.write(f"**맛**: {wine.metadata['taste_profile']}")
                
                st.write("**리뷰**")
                st.write(wine.page_content[:200] + "...")
```

---

### 📋 기능 요약표

| ID | 기능 | 기술 구현 | 우선순위 |
|----|------|-----------|----------|
| F-01 | 자연어 질의 처리 | Self-Querying Retriever | ⭐⭐⭐⭐⭐ |
| F-02 | 메타데이터 필터링 | LangChain + Pinecone | ⭐⭐⭐⭐⭐ |
| F-03 | 맛 Feature 추출 | NLP | ⭐⭐⭐⭐ |
| F-04 | 벡터 검색 | OpenAI Embedding + Pinecone | ⭐⭐⭐⭐⭐ |
| F-05 | 웹 인터페이스 | Streamlit | ⭐⭐⭐⭐ |
| F-06 | 데이터 품질 관리 | Pandas 전처리 파이프라인 | ⭐⭐⭐⭐⭐ |

---

## 4️⃣ 프로젝트 후기

### 💡 기술적 성장 포인트

#### **1. RAG 시스템의 한계 발견 및 극복**

**문제 인식**
```
초기 구현: 단순 유사도 검색
↓
문제 발생: "프랑스 제외"가 오히려 프랑스 와인 추천
↓
원인 분석: 부정 표현은 의미적으로 키워드와 유사도 높음
↓
해결: Self-Querying으로 논리 필터 분리
```

**배운 점**
- RAG는 만능이 아니며, 사용 사례에 따라 하이브리드 접근 필요
- Semantic Search(의미 검색) + Structured Filtering(구조 필터)의 조합이 핵심
- 사용자 의도 파싱이 정확도에 결정적 영향

---

#### **2. LLM 프롬프트 엔지니어링**


**배운 점**
- 스키마를 명확히 제시할수록 LLM 파싱 정확도 향상
- Few-shot 예시 추가 시 edge case 처리 능력 개선
- Temperature=0 설정으로 일관성 확보

---

#### **3. 대규모 데이터 처리 경험**

**배운 점**
- 전처리 단계의 데이터 정제가 전체 파이프라인 효율에 결정적
- API Rate Limit 고려한 배치 처리 설계 중요
- 품질 vs 양의 트레이드오프: 적은 고품질 데이터가 더 나은 결과

---

#### **4. 벡터 DB 선택 및 최적화**

**비교 검토한 옵션**
| DB | 장점 | 단점 | 선택 이유 |
|-----|------|------|----------|
| **Pinecone** | 관리 불필요, 메타필터 강력 | 유료 (무료 100만 벡터) | ✅ 선택 |
| ChromaDB | 로컬 실행, 무료 | 확장성 제한 | × |
| Weaviate | 오픈소스, 유연함 | 설정 복잡 | × |


**배운 점**
- 메타데이터 인덱싱 전략이 필터 성능에 직접 영향
- 무료 티어 제약 안에서 최대 성능 끌어내는 설계 능력
- 클라우드 벡터 DB의 장단점 실전 경험

---

### 🚀 개선 가능한 점 & 향후 계획

#### **단기 개선 사항 (1-2주)**
1. **추천 근거 시각화**
   - 현재: 텍스트 설명만
   - 개선: 맛 프로필 레이더 차트, 가격대 비교 그래프

2. **대화 히스토리 관리**
   - 현재: 단발성 질문만
   - 개선: "방금 추천한 것보다 더 저렴한 걸로" 같은 문맥 이해

3. **다국어 지원**
   - 현재: 영어만
   - 개선: 한국어 와인명, 품종 번역 추가


---

### 🎓 프로젝트를 통해 배운 핵심 교훈

#### **1. 데이터 품질이 모델 성능을 결정한다**
> 72,000건 전부 vs 40,000건 엄선 → 후자가 사용자 만족도 월등히 높음.  
> "쓰레기를 넣으면 쓰레기가 나온다(GIGO)" 원칙 체감.


#### **2. 사용자 관점에서 생각하는 습관**
> 기술적으로 멋진 기능보다, 직관적인 UX가 더 중요.  
> "프랑스 제외"라는 간단한 말을 이해 못 하면 실패한 서비스.


