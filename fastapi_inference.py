# fastapi_inference.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoConfig
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel

app = FastAPI(title="친밀도 모델 로컬 추론 API")

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
        intimacy_score = self.sigmoid(intimacy_score) * 5  # 0~5 범위
        return {"intimacy_scores": intimacy_score}

# 모델 및 토크나이저 로드
MODEL_NAME = "kelly9457/bindly-I-v4"  # 업로드된 친밀도 모델
config = AutoConfig.from_pretrained(MODEL_NAME, num_labels=1, problem_type="regression")
model = KoBERTIntimacy.from_pretrained(MODEL_NAME, config=config)
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# 요청 데이터 형식 정의
class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_intimacy(data: TextInput):
    text = data.text
    if not text:
        raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")
    
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    score = outputs["intimacy_scores"].item()  # 0~5 범위
    scaled_score = score * 20  # 0~100으로 변환pip install fastapi uvicorn torch transformers

    return {"raw_score": score, "scaled_score": scaled_score}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)