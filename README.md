# ⚡ Bindly FastAPI
<h4 align="center"> 
<img src="https://github.com/user-attachments/assets/acbce417-7713-4fef-87db-602a7c25191c" alt="long" border="0">
</h4>
 
**Bindly** 프로젝트의 모델 예측 수행을 담당하는 FastAPI 기반 저장소입니다. ⚡

----
## 🚀 프로젝트 개요
> Bindly FastAPI는 다음과 같은 기능을 제공합니다:

<details>
<summary> 💑 대화 상대와의 친밀도 분석 </summary>
카카오톡 대화 기록을 기반으로 친밀도를 평가하여 점수화합니다.
</details>
 
<details>
<summary> 📊 대화 스타일 및 관계 예측 </summary>
대화 패턴을 분석하여 사용자의 대화 스타일(공감형)과, 상대방과의 관계를 예측해 줍니다.
</details>

<details>
<summary> 🔍 KLUE-BERT 모델 기반 감정 분석 </summary>
대화 데이터를 분석하여 긍정적인 대화 흐름과, 부정적인 대화 흐름을 판별합니다.
</details>

<details>
<summary> 🤖 AI에게 받는 피드백 </summary>
OpenAI와 연결하여, 사용자의 대화 습관을 피드백받고, 개선된 대화 예시를 보여 줍니다.
</details>

<details>
<summary> ⚡ 고속 API 응답 </summary>
FastAPI를 기반으로 빠르고 확장 가능한 API를 제공합니다.
</details>

## 🤗 허깅페이스에 서빙 중인 Bindly AI 모델
- [친밀도 예측](https://huggingface.co/kelly9457/bindly-I-v4) 모델
- [관계 예측](https://huggingface.co/kelly9457/bindly-R) 모델
- [긍정, 부정적 대화 흐름 판별](https://huggingface.co/chihopark/bindly-sentiment-v6) 모델

 *저희 모델은 모두 [KLUE/BERT-BASE 모델](https://huggingface.co/klue/bert-base)을 기반으로 파인튜닝 하였습니다. 🔧*
  
## 📂 디렉토리 구조

```
📦 bindly_fastapi
├── 📜 main.py              # FastAPI 서버 실행 엔트리포인트
├── 📜 fastapi_inference.py  # 모델 추론을 위한 API 엔드포인트
├── 📜 tokenize_inference.py # 입력 데이터를 토크나이즈하는 모듈
├── 📂 models               # 모델 관련 파일 폴더
│   ├── 📜 empathy_mean_vector.pth  # 공감 모델 평균 벡터
│   ├── 📜 vocab.txt               # 모델의 어휘 사전
├── 📂 utils                # 유틸리티 모음
│   ├── 📜 helpers.py              # 보조 함수 정의
├── 📂 tests                # 테스트 코드 폴더
│   ├── 📜 test_api.py            # API 엔드포인트 테스트
├── 📜 requirements.txt     # 프로젝트 의존성 목록
├── 📜 .gitignore           # Git에서 제외할 파일 목록
└── 📜 README.md            # 프로젝트 문서
```

## ⚙️ 설치 및 실행 방법

1️⃣ **저장소 클론**
   ```bash
   git clone https://github.com/NeedTalkKey/bindly_fastapi.git
   cd bindly_fastapi
   ```

2️⃣ **가상 환경 생성 및 활성화**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows의 경우 `venv\Scripts\activate`
   ```

3️⃣ **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

4️⃣ **FastAPI 서버 실행**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

   서버는 기본적으로 `http://127.0.0.1:8000`에서 실행됩니다.

## 🔍 사용 방법

서버가 실행되면, `/docs` 엔드포인트에서 자동 생성된 Swagger UI를 통해 API를 테스트하고 문서를 확인할 수 있습니다.

## 🤝 기여 방법

1. 이 저장소를 포크합니다.
2. 새로운 브랜치를 생성합니다: `git checkout -b feature/새로운기능`
3. 변경 사항을 커밋합니다: `git commit -m '새로운 기능 추가'`
4. 브랜치에 푸시합니다: `git push origin feature/새로운기능`
5. Pull Request를 생성합니다.

## 📜 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

---

**Bindly** 프로젝트의 다른 Git도 둘러보세요!
> 🐇 [Bindly Front-end](https://github.com/NeedTalkKey/bindly_front)
> 
> 🐰 [Bindly Back-end](https://github.com/NeedTalkKey/bindly_back)

