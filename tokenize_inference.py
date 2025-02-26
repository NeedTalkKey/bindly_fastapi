from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer
import math

app = FastAPI(title="토큰화 및 청킹 API")

# 사용하고자 하는 모델에 맞는 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("klue/bert-base")

# 요청 데이터 형식 정의
class TokenizeInput(BaseModel):
    text: str
    max_length: int = 300  # 각 청크의 최대 토큰 수
    overlap: int = 50      # 청크 간 오버랩 토큰 수

@app.post("/tokenize")
def tokenize_text(data: TokenizeInput):
    text = data.text
    if not text:
        raise HTTPException(status_code=400, detail="텍스트를 입력해주세요.")
    
    # 토큰화 (특수 토큰 제외)
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    total_tokens = len(token_ids)
    chunks = []
    start = 0
    
    # 청크 생성: max_length와 overlap을 고려
    while start < total_tokens:
        chunk_ids = token_ids[start : start + data.max_length]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
        chunks.append(chunk_text)
        # 다음 청크의 시작 위치: overlap 만큼 겹치게
        start += (data.max_length - data.overlap)
    
    return {"chunks": chunks, "total_tokens": total_tokens, "num_chunks": len(chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
