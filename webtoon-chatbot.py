import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.document_loaders import CSVLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

import re
from pathlib import Path

import sqlite3
from datetime import datetime

from langchain.chains import RetrievalQA
from langchain.memory import ConversationSummaryBufferMemory

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
####################################################################################

# OpenAI API Key 등록
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 없습니다. .env에 설정해 주세요.")

# LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024)

# DB 경로
WEBTOON_CSV_PATH = "./webtoon_data.csv" # 웹툰 통계 벡터DB
CHROMA_DIR = "./chroma_db5"      # 법률/가이드 문서 벡터DB
FAISS_DB_DIR = "./db"            # webtoon_synopsis.csv 기반 FAISS 디렉토리

@st.cache_resource
def build_or_load_vectorstores():
    # ---- 1) CSV 로드 (필수 컬럼 유효성 점검) ----
    df = pd.read_csv(WEBTOON_CSV_PATH)
    if "조회수" in df.columns and "구독자수" not in df.columns:
        df = df.rename(columns={"조회수": "구독자수"})
    # 안전한 기본 컬럼 체크
    for col in ["제목", "평점", "구독자수"]:
        if col not in df.columns:
            st.warning(f"CSV에 '{col}' 컬럼이 없습니다. 일부 기능이 제한될 수 있어요.")

    # ---- 2) 법률/가이드 문서용 Chroma ----
    # 없으면 그냥 빈 DB여도 되지만, 최소 폴더 생성
    Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
    embeddings_hf = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
    vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings_hf)
    retriever_law = vectordb.as_retriever(search_kwargs={"k": 10})
    rag_chain_law = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever_law)

    # ---- 3) 웹툰 정보용 FAISS (없으면 CSV에서 즉시 빌드) ----
    faiss_path = Path(FAISS_DB_DIR)
    if faiss_path.exists() and any(faiss_path.iterdir()):
        embedding_info = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore_info = FAISS.load_local(FAISS_DB_DIR, embedding_info, allow_dangerous_deserialization=True)
    else:
        # CSV 각 행을 하나의 문서로 임베딩하여 정보 검색 가능하게 구성
        records = []
        for _, row in df.iterrows():
            title = str(row.get("제목", ""))
            summary = str(row.get("줄거리", row.get("키워드", "")))
            meta = {k: (None if pd.isna(v) else v) for k, v in row.to_dict().items()}
            content = f"제목: {title}\n요약: {summary}\n메타: {meta}"
            records.append(content)
        embedding_info = OpenAIEmbeddings(model="text-embedding-3-small")
        vectorstore_info = FAISS.from_texts(records, embedding_info)
        vectorstore_info.save_local(FAISS_DB_DIR)

    retriever_info = vectorstore_info.as_retriever(search_kwargs={"k": 10})
    memory_info = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    return df, rag_chain_law, vectordb, retriever_info, memory_info

df, rag_chain, vectordb, retriever_info, memory_info = build_or_load_vectorstores()

# ---- Intent/파싱 고도화 ----
prompt_template_info = ChatPromptTemplate.from_messages([
    ("system", "너는 웹툰 전문가야. 반드시 웹툰 데이터베이스(검색 문서) 내 정보만 사용해. \
문서가 없으면 '해당 정보를 찾을 수 없습니다'라고만 말해."),
    ("human", "질문: {question}\n\n관련 문서:\n{context}")
])

def custom_rag_chatbot(query, retriever, llm, memory):
    docs = retriever.get_relevant_documents(query)
    if not docs:
        return "해당 정보를 웹툰 데이터베이스에서 찾을 수 없습니다. 웹툰과 관련된 다른 질문을 해주세요."
    context = "\n\n".join([d.page_content for d in docs])
    messages = prompt_template_info.format_messages(question=query, context=context)
    response = llm(messages)
    memory.chat_memory.add_user_message(query)
    memory.chat_memory.add_ai_message(response.content)
    return response.content

# 의도 분류를 구조화 출력으로 강제
def classify_intent(question: str) -> str:
    schema = """다음 중 하나만 출력: 법률 | 정보 | 현황 | 추천
규칙:
- '법률': 2차 창작(드라마/영화/게임/애니 등) 시의 법률/저작권/계약 등 법적 이슈
- '정보': 제목/줄거리/작가/등장인물 등의 작품 자체 정보
- '추천': 2차 창작 용도로 적합한 웹툰 추천 요구 (2차 창작 언급 없으면 추천 아님)
- '현황': 조회수/구독자수/평점/순위/선호도 등 통계/랭킹
질문: """ + question + "\n정답:"
    ans = llm.predict(schema).strip()
    if ans not in {"법률", "정보", "현황", "추천"}:
        # fallback
        if any(k in question for k in ["조회수", "구독자수", "평점", "순위", "선호도"]):
            return "현황"
        if any(k in question for k in ["추천", "골라줘", "뭐가 좋아"]):
            return "추천"
        return "정보"
    return ans

def extract_top_k(question: str, default_k=10) -> int:
    # 숫자 + (개|편|명|위|top) 패턴 다양화
    m = re.search(r"(?:상위|top\s*)?(\d+)\s*(?:개|편|명|위)?", question, re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
            return max(1, min(20, n))
        except:
            pass
    return default_k

def extract_filter_conditions(question: str) -> dict:
    prompt = f"""질문에서 아래 조건을 추출.
- 카테고리: 장르 (로맨스/무협/드라마/액션/스릴러 등). 모르면 빈칸.
- 연령층: 10~20대 / 30대 / 40대 중 택1. 모르면 빈칸.
- 성별: 남성 / 여성 중 택1. 모르면 빈칸.

출력 형식(그대로):
카테고리:
연령층:
성별:

질문: "{question}"
"""
    result = llm.predict(prompt).strip()
    cond = {"카테고리": "", "연령층": "", "성별": ""}
    for line in result.splitlines():
        if ":" in line:
            k, v = [s.strip() for s in line.split(":", 1)]
            if k in cond:
                cond[k] = v
    return cond

def _safe_contains(series: pd.Series, needle: str) -> pd.Series:
    if not needle:
        return pd.Series([True]*len(series), index=series.index)
    return series.fillna("").astype(str).str.contains(needle, case=False, na=False)

def handle_recommendation(question: str) -> str:
    k = extract_top_k(question, default_k=3)
    cond = extract_filter_conditions(question)
    filtered = df.copy()

    if "카테고리" in df.columns and cond["카테고리"]:
        filtered = filtered[_safe_contains(filtered["카테고리"], cond["카테고리"])]

    # 연령층/성별 관련 컬럼이 없는 경우 우회
    if cond["연령층"] in df.columns:
        filtered = filtered[_safe_contains(filtered[cond["연령층"]], "상|중")]
    if "성별선호도" in df.columns and cond["성별"]:
        filtered = filtered[_safe_contains(filtered["성별선호도"], cond["성별"])]

    if filtered.empty:
        return "❗ 조건에 맞는 웹툰을 찾지 못했습니다."

    # 간단 점수식 (가중치 상수는 설정값화 가능)
    score = []
    for _, row in filtered.iterrows():
        rating = float(row.get("평점", 0) or 0)
        subs = float(row.get("구독자수", 0) or 0)
        score.append(rating * 1.0 + (subs / 1000.0))
    filtered = filtered.assign(추천점수=score).sort_values("추천점수", ascending=False).head(k)

    lines = []
    for _, r in filtered.iterrows():
        title = r.get("제목", "N/A")
        rating = r.get("평점", "N/A")
        subs = int(float(r.get("구독자수", 0)))
        kw = r.get("키워드", "")
        lines.append(f"- {title} (평점: {rating}, 구독자수: {subs}, 키워드: {kw})")
    return "\n".join(lines)

def handle_status(question: str) -> str:
    k = extract_top_k(question, default_k=3)
    cond = extract_filter_conditions(question)
    filtered = df.copy()

    if "카테고리" in df.columns and cond["카테고리"]:
        filtered = filtered[_safe_contains(filtered["카테고리"], cond["카테고리"])]
    if cond["연령층"] in df.columns:
        filtered = filtered[_safe_contains(filtered[cond["연령층"]], "상|중")]
    if "성별선호도" in df.columns and cond["성별"]:
        filtered = filtered[_safe_contains(filtered["성별선호도"], cond["성별"])]

    # 정렬 기준 자동 판별
    if "구독자수" in question:
        key = "구독자수"
    elif "평점" in question:
        key = "평점"
    else:
        key = None

    if key and key in filtered.columns:
        top = filtered.sort_values(key, ascending=False).head(k)
    else:
        cols = [c for c in ["평점", "구독자수"] if c in filtered.columns]
        if cols:
            top = filtered.sort_values(cols, ascending=False).head(k)
        else:
            return "❗ 조건에 맞는 웹툰이 없습니다."

    lines = []
    for _, r in top.iterrows():
        title = r.get("제목", "N/A")
        rating = r.get("평점", "N/A")
        subs = int(float(r.get("구독자수", 0)))
        cat = r.get("카테고리", "")
        lines.append(f"- {title} (평점: {rating}, 구독자수: {subs}, 카테고리: {cat})")
    return "\n".join(lines)



# Streamlit UI
st.title("📚 웹툰 통합 챗봇 (법률 + 정보 + 현황 + 추천)")
st.caption("2차 창작 + 데이터 통계 + 웹툰 정보 제공 챗봇")

if "history" not in st.session_state:
    st.session_state.history = []

with st.form("input_form", clear_on_submit=True):
    user_input = st.text_input("질문을 입력하세요:", placeholder="예: 30대 여성이 좋아할 드라마 웹툰 추천해줘")
    submitted = st.form_submit_button("질문하기")

if submitted and user_input:
    intent = classify_intent(user_input)
    response = ""
    chunks = []

    if intent == "법률":
        result = rag_chain(user_input)
        response = result["answer"]
        chunks = vectordb.similarity_search(user_input, k=3)
    elif intent == "현황":
        response = handle_status(user_input)
    elif intent == "추천":
        response = handle_recommendation(user_input)
    elif intent == "정보":
        response = custom_rag_chatbot(user_input, retriever_info, llm, memory_info)
    else:
        response = "❗ 질문을 이해하지 못했습니다. 다시 시도해주세요."

    st.session_state.history.append({
        "질문": user_input,
        "답변": response,
        "intent": intent,
        "chunks": chunks
    })

for chat in reversed(st.session_state.history):
    st.markdown(f"**💬 질문:** {chat['질문']}")
    if chat["intent"] == "법률":
        st.markdown("**📚 법률 응답:**")
    elif chat["intent"] == "현황":
        st.markdown("**📊 현황 응답:**")
    elif chat["intent"] == "추천":
        st.markdown("**🎯 추천 응답:**")
    elif chat["intent"] == "정보":
        st.markdown("**📘 웹툰 정보 응답:**")
    st.markdown(chat['답변'].replace("\n", "  \n"))
    st.divider()