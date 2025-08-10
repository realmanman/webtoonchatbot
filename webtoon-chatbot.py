import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma, FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate

from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from datetime import datetime, timezone
from pathlib import Path
import json, csv

import re

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
    _KST = ZoneInfo("Asia/Seoul")
except Exception:
    _KST = timezone.utc  # 폴백 (UTC)
####################################################################################

# OpenAI API Key 등록
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("환경변수 OPENAI_API_KEY가 없습니다. .env에 설정해 주세요.")

# 본문 답변용 LLM 설정
llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=1024)

# 분류/추출 전용: 함수콜(Structured Output) LLM
llm_struct = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=256)

# Intent 결과 스키마 정의 (로그/분석)
# ---------- Pydantic Schemas ----------
class IntentResult(BaseModel):
    intent: Literal["법률", "정보", "현황", "추천"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: str

class FilterResult(BaseModel):
    # 비어 있을 수도 있으니 Optional[str]
    카테고리: Optional[str] = ""
    연령층: Optional[Literal["10~20대", "30대", "40대", ""]] = ""
    성별: Optional[Literal["남성", "여성", ""]] = ""

# Pydantic으로 직접 파싱되는 래퍼 (툴콜 사용)
intent_llm = llm_struct.with_structured_output(IntentResult)
filter_llm = llm_struct.with_structured_output(FilterResult)

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
    retriever_law = vectordb.as_retriever(search_kwargs={"k": 5})
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

    retriever_info = vectorstore_info.as_retriever(search_kwargs={"k": 5})
    memory_info = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")

    return df, rag_chain_law, vectordb, retriever_info, memory_info

df, rag_chain, vectordb, retriever_info, memory_info = build_or_load_vectorstores()

# ---------- 프롬프트 ----------
intent_prompt = ChatPromptTemplate.from_messages([
    ("system",
     """너는 사용자 질문의 의도를 다음 중 하나로 정확히 분류한다.
규칙:
- "법률": 2차 창작(드라마/영화/게임/애니 등) 관련 저작권/계약/법령/법적 이슈 문의
- "정보": 작품 자체 정보(제목/줄거리/작가/등장인물/설정 등)
- "추천": 2차 창작 목적의 '적합한 웹툰 추천' 요청 (2차 창작 언급 없으면 추천 아님)
- "현황": 조회수/구독자수/평점/순위/연령·성별 선호도 등 통계·랭킹 요청
"""
),
    ("human", "질문: {question}\n하나의 라벨을 고르고, 신뢰도와 간단한 근거를 함께 내.")
])

filters_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "사용자 질문에서 추천/현황 분석에 필요한 필터를 추출한다. "
     "카테고리(자유 텍스트), 연령층(10~20대/30대/40대/빈), 성별(남성/여성/빈)을 채워라."),
    ("human", "질문: {question}")
])

def _strip_code_fences(text: str) -> str:
    t = text.strip()
    # ```json ... ``` or ``` ... ```
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, flags=re.IGNORECASE)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()

def _regex_salvage(text: str) -> dict | None:
    """
    키 따옴표가 망가진 경우까지 긁어오기 위한 마지막 수단.
    intent|confidence|reasons 키를 관대하게 탐지.
    """
    t = _strip_code_fences(text)
    # 중괄호 블록만 먼저 추출
    m = re.search(r'\{.*\}', t, flags=re.DOTALL)
    if m:
        t = m.group(0)
    # 키:값 패턴 느슨하게 매칭 (따옴표 유무 허용)
    kv = {}
    for key in ["intent", "confidence", "reasons"]:
        # 예: "''intent''" : "추천"  또는  intent: 추천
        p = rf'["\'\s]*{key}["\'\s]*\s*:\s*(".*?"|\d+(\.\d+)?|true|false|null|[^,}}]+)'
        mm = re.search(p, t, flags=re.IGNORECASE|re.DOTALL)
        if mm:
            rawv = mm.group(1).strip()
            # 값에서 감싼 따옴표 제거
            if rawv.startswith('"') and rawv.endswith('"'):
                rawv = rawv[1:-1]
            kv[key] = rawv
    return kv or None

def _normalize_quoted_keys(obj: dict) -> dict:
    norm = {}
    for k, v in obj.items():
        kk = str(k)
        # 앞뒤 중복 따옴표/공백 제거  e.g. "'\"intent\"'" → intent
        kk = re.sub(r'^[\s\'"]+|[\s\'"]+$', "", kk)
        norm[kk] = v
    return norm

def tolerant_pydantic_parse(text: str, model_cls: type[BaseModel]):
    # 0) 그대로
    for candidate in [text, _strip_code_fences(text)]:
        try:
            return model_cls.parse_obj(json.loads(candidate))
        except Exception:
            pass
        # 단일따옴표 → 이중따옴표
        try:
            t2 = candidate.replace("'", '"')
            obj = json.loads(t2)
            obj = _normalize_quoted_keys(obj)
            return model_cls.parse_obj(obj)
        except Exception:
            pass

    # 1) ""intent"" → "intent" 교정
    t3 = re.sub(r'"{2,}\s*([A-Za-z_][\w\- ]*)\s*"{2,}\s*:', r'"\1":', text)
    try:
        obj = json.loads(_strip_code_fences(t3))
        obj = _normalize_quoted_keys(obj)
        return model_cls.parse_obj(obj)
    except Exception:
        pass

    # 2) 정규식 구조화(최후 수단)
    salvaged = _regex_salvage(text)
    if salvaged:
        try:
            # 결측 보정
            if "confidence" in salvaged:
                try:
                    salvaged["confidence"] = float(salvaged["confidence"])
                except Exception:
                    salvaged["confidence"] = 0.0
            salvaged.setdefault("reasons", "")
            # intent 라벨 정규화
            if "intent" in salvaged:
                lbl = salvaged["intent"].strip()
                # 오타/영문 대응
                mapping = {
                    "법률":"법률", "legal":"법률",
                    "정보":"정보", "info":"정보",
                    "현황":"현황", "status":"현황", "통계":"현황",
                    "추천":"추천", "recommend":"추천"
                }
                salvaged["intent"] = mapping.get(lbl.lower(), lbl)
            return model_cls.parse_obj(salvaged)
        except Exception:
            pass

    return None

# ---------- 의도 분류 + 로그 저장 ----------
INTENT_LOG_PATH = Path("./intent_logs.csv")

def _now_kst_str() -> str:
    return datetime.now(tz=_KST).strftime("%Y-%m-%d %H:%M:%S%z")

def log_intent_row(question: str, res: IntentResult, raw_payload: str = ""):
    write_header = not INTENT_LOG_PATH.exists()
    with INTENT_LOG_PATH.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["timestamp_kst","question","intent","confidence","reasons","raw"])
        w.writerow([_now_kst_str(), question, res.intent, f"{res.confidence:.3f}", res.reasons, raw_payload])

def _rule_based_fallback(q: str, default: str = "정보") -> str:
    ql = q.lower()
    law_kw = ["저작권","법","법률","계약","초상권","허가","라이선스","드라마화","영화화","각색","판권","ip","리메이크","원작 계약","섭외"]
    status_kw = ["조회수","구독자수","평점","랭킹","순위","선호도","통계"]
    rec_kw = ["추천","골라줘","픽","고르면","추천해","골라","뭘 볼까","추천 좀"]
    if any(k in q for k in law_kw): return "법률"
    if any(k in q for k in status_kw): return "현황"
    if ("드라마" in q or "영화" in q or "게임" in q or "애니" in q) and any(k in ql for k in rec_kw): return "추천"
    if any(k in ql for k in rec_kw): return "추천"
    return default

# ---------- 분류 ----------
def classify_intent(question: str, confidence_threshold: float = 0.5) -> str:
    # 1) 함수콜 기반 Structured Output (파싱 실패 거의 없음)
    msgs = intent_prompt.format_messages(question=question)
    res: IntentResult = intent_llm.invoke(msgs)
    intent = res.intent
    conf = max(0.0, min(1.0, float(res.confidence)))

    # 2) 약하면 룰 보정
    if conf < confidence_threshold:
        intent = _rule_based_fallback(question, default=intent)

    # 3) 원문 로그: 함수콜에선 raw JSON이 없으니, 모델 입력(질문)과 도출값을 JSON처럼 기록
    raw_payload = f'{{"intent":"{res.intent}", "confidence":{res.confidence}, "reasons":"{res.reasons}"}}'
    log_intent_row(question, res, raw_payload=raw_payload)
    return intent

def extract_filters_structured(question: str) -> dict:
    raw = ""
    try:
        msgs = filters_prompt.format_messages(
            format_instructions=filter_parser.get_format_instructions(),
            question=question,
        )
        resp = llm_struct.invoke(msgs)  # ✅ JSON 강제
        raw = resp.content or ""

        try:
            parsed: FilterResult = filter_parser.parse(raw)
        except Exception:
            parsed = tolerant_pydantic_parse(raw, FilterResult)
            if parsed is None:
                raise KeyError('filters_parse_failed')

        return parsed.dict()

    except Exception:
        # 폴백
        cond = {"카테고리": "", "연령층": "", "성별": ""}
        cats = ["로맨스", "드라마", "액션", "무협", "스릴러", "판타지", "코미디", "학원", "스포츠", "공포"]
        for c in cats:
            if c in question:
                cond["카테고리"] = c
                break
        if any(k in question for k in ["10대", "20대", "10~20대", "젊은", "청소년", "Z세대"]):
            cond["연령층"] = "10~20대"
        elif "30대" in question or "직장인" in question:
            cond["연령층"] = "30대"
        elif "40대" in question:
            cond["연령층"] = "40대"
        if "여성" in question or "여자" in question:
            cond["성별"] = "여성"
        elif "남성" in question or "남자" in question:
            cond["성별"] = "남성"
        return cond        

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

def extract_top_k(question: str, default_k=5) -> int:
    # 숫자 + (개|편|명|위|top) 패턴 다양화
    m = re.search(r"(?:상위|top\s*)?(\d+)\s*(?:개|편|명|위)?", question, re.IGNORECASE)
    if m:
        try:
            n = int(m.group(1))
            return max(1, min(5, n))
        except:
            pass
    return default_k

def _safe_contains(series: pd.Series, needle: str) -> pd.Series:
    if not needle:
        return pd.Series([True]*len(series), index=series.index)
    return series.fillna("").astype(str).str.contains(needle, case=False, na=False)

def handle_recommendation(question: str) -> str:
    k = extract_top_k(question, default_k=5)
    cond = extract_filters_structured(question)
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
    k = extract_top_k(question, default_k=5)
    cond = extract_filters_structured(question)
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
        chunks = vectordb.similarity_search(user_input, k=5)
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