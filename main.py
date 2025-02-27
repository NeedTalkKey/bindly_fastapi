from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoConfig
import torch.nn as nn
from transformers import BertPreTrainedModel, BertTokenizer, BertModel
import torch.nn.functional as F
import math
from typing import Dict

app = FastAPI(title="통합 FastAPI Inference 서버")

# CORS 설정: 모든 출처, 메서드, 헤더 허용 (필요에 따라 조정)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 특정 도메인만 허용하려면 여기에 추가하세요.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

##############################
# 1. 토큰화 및 청킹 엔드포인트
##############################

class TokenizeInput(BaseModel):
    text: str
    max_length: int = 300   # 각 청크 당 최대 토큰 수
    overlap: int = 50       # 청크 간 중복 토큰 수

# 토큰화 전용 토크나이저 (예: "klue/bert-base")
tokenizer_tokenize = AutoTokenizer.from_pretrained("klue/bert-base")

@app.post("/tokenize")
def tokenize_text(data: TokenizeInput):
    text = data.text
    if not text:
        raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")
    token_ids = tokenizer_tokenize.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    chunks = []
    start = 0
    while start < total_tokens:
        chunk_ids = token_ids[start : start + data.max_length]
        chunk_text = tokenizer_tokenize.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        start += (data.max_length - data.overlap)
    return {"chunks": chunks, "total_tokens": total_tokens, "num_chunks": len(chunks)}

################################
# 2. 친밀도 모델 예측 엔드포인트
################################

# 모델 클래스 정의 (KoBERTIntimacy)
class KoBERTIntimacy(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.intimacy_regressor = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled_output = outputs.pooler_output
        intimacy_score = self.intimacy_regressor(pooled_output).squeeze(1)
        # 0~5 범위로 정규화
        intimacy_score = self.sigmoid(intimacy_score) * 5  
        return {"intimacy_scores": intimacy_score}

# 모델 및 토크나이저 로드
MODEL_NAME = "kelly9457/bindly-I-v4"  # 친밀도 모델 이름 (적절히 수정)
config_model = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1, problem_type="regression")
model = KoBERTIntimacy.from_pretrained(MODEL_NAME, config=config_model)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer_predict = AutoTokenizer.from_pretrained(MODEL_NAME)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_intimacy(data: TextInput):
    text = data.text
    if not text:
        raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")
    inputs = tokenizer_predict(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    score = outputs["intimacy_scores"].item()  # 0~5 범위
    scaled_score = score * 20  # 0~100으로 변환
    return {"raw_score": score, "scaled_score": scaled_score}

class SpeakerModel(BaseModel):
    u1: str
    u2: str

@app.post("/empathy")
def empathy_measure(corpus: str, speaker: SpeakerModel):
    lines = corpus.split('\n')
    u1 = []
    u2 = []
    for idx, text in enumerate(lines):
        startIdx = len(text.split(": ")[0]) + 2
        text = text[startIdx:]

        # u1, u2의 문장을 각각 리스트로 담음
        if idx % 2 == 0:
            u1.append(text)
        else:
            u2.append(text)

    # [{"speaker": "건우", "score":60}, {"speaker": "남희", "score":70}]
    dict1 = {"speaker" : speaker.u1, "score": textEmpathyMesureAvg(u1)}
    dict2 = {"speaker": speaker.u1, "score": textEmpathyMesureAvg(u2)}

    return [dict1, dict2]

# parameter : [문장1, 문장2, 문장3, ...]
# return : 60%    등 평균 공감 점수
def textEmpathyMesureAvg(lines: list):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        mean_embedding = torch.load("model/empathy_mean_vector.pth", map_location=device)
        print("✅ 공감형 평균 벡터 로드 완료")
    except Exception as e:
        print("❌ 평균 벡터 로드 실패:", e)
        return None
    tokenizer = BertTokenizer.from_pretrained("klue/bert-base")
    model = BertModel.from_pretrained("klue/bert-base")
    model.to(device)
    model.eval()
    print("✅ 모델 및 토크나이저 로드 완료")

    total_confidence = 0
    for parameter_text in lines:
        predicted_label, confidence = predict_empathy(parameter_text, mean_embedding, tokenizer)
        total_confidence += math.floor(confidence * 100)
    return math.floor(total_confidence / len(lines))


# 입력 문장의 특징 벡터 추출
def get_embedding(text, tokenizer):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding='max_length', max_length=512)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1)

# 유사도 기반 분류 함수
def predict_empathy(text, mean_embedding, tokenizer):
    print("🔍 예측 실행 중...")  # 디버깅용
    text_embedding = get_embedding(text, tokenizer)

    # 코사인 유사도 계산 (공감형 평균 벡터와 비교)
    similarity = F.cosine_similarity(text_embedding, mean_embedding.unsqueeze(0))

    # ✅ 유사도 평균 계산 (문장 전체 유사도)
    similarity_score = similarity.mean().item()

    # 임계값 설정 (유사도가 0.5 이상이면 '공감형', 아니면 '직설형')
    label = "공감형" if similarity_score >= 0.5 else "직설형"

    print(f"✅ 예측 완료: {label} (유사도: {similarity_score:.4f})")  # 결과 출력
    return label, similarity_score

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
