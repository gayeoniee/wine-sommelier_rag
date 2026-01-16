import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import streamlit as st

load_dotenv()

@st.cache_resource
def get_wine_rag_chain():
    # 1. 초기화
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    # 2. 벡터 스토어 연결
    vectorstore = PineconeVectorStore(
        index_name="wine-sommelier-agent", 
        embedding=embeddings
    )

    # 3. 셀프 쿼링 설정
    metadata_field_info = [
        AttributeInfo(
            name="country",
            description="The country where the wine was produced. For European wines, use: France, Italy, Spain, Portugal, Germany, Austria, Greece. For American wines, use: US. For Oceania: Australia, New Zealand. For South American wines: Chile, Argentina.",
            type="string"
        ),
        AttributeInfo(name="points", description="Wine rating score from 80 to 100", type="integer"),
        AttributeInfo(name="price", description="Price in USD", type="float"),
        AttributeInfo(name="variety", description="Grape variety like Chardonnay, Pinot Noir, Cabernet Sauvignon", type="string"),
        AttributeInfo(name="tag_oak", description="1 if wine has oak aging flavor, else 0", type="integer"),
    ]
    document_content_description = "Wine tasting notes, flavor profile, and characteristics description"

    retriever = SelfQueryRetriever.from_llm(
        llm, vectorstore, document_content_description, metadata_field_info, verbose=True
    )

    # 4. RAG 프롬프트 및 체인
    template = """당신은 데이터 기반 전문 소믈리에입니다.
    반드시 아래의 [출력 양식]을 엄격히 준수하여 답변하세요.

    **중요한 규칙:**
    1. 와인 이름은 **반드시 영어 원문**으로 표기하세요 (예: Château Margaux, not 샤토 마고)
    2. 전문가 노트, 선정 이유, 페어링 제안 등 **모든 설명은 한글**로 작성하세요
    3. 사용자가 "유럽 와인"을 요청하면 France, Italy, Spain, Portugal, Germany 등의 유럽 국가만 선택하세요
    4. 사용자가 "미국 와인"을 요청하면 반드시 country가 "US"인 와인만 선택하세요
    5. 사용자가 특정 대륙/지역을 언급하면 해당 지역의 와인만 추천하세요

    ### [출력 양식]
    ---
    #### 🍷 추천 와인: [와인 영어 이름 및 빈티지]
    - **산지/품종:** [국가명] | [포도 품종]
    - **데이터 분석:** 평점 **[XX점]** / 가격 **$[XX]** (가성비 지수: [우수/보통])
    - **전문가 노트:** [맛과 향에 대한 핵심 설명 2문장 이내 - 반드시 한글로]
    - **선정 이유:** [사용자의 요청과 연계된 논리적 추천 근거 - 반드시 한글로]
    - **🍴 페어링 제안:** [어울리는 음식 1~2가지 - 반드시 한글로]
    ---

    검색된 와인 정보:
    {context}

    사용자 질문: {question}

    답변:"""
    prompt = ChatPromptTemplate.from_template(template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain