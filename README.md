# Transformer PoC (Proof of Concept) 구현 내용

## 1. 개요

본 프로젝트는 Transformer 아키텍처의 핵심 구성 요소인 인코더 블록(Encoder Block)을 PyTorch를 사용하여 모듈화하여 구현한 개념 증명(PoC)입니다. 각 기능을 별도의 Python 모듈로 분리하여 코드의 재사용성과 가독성을 높이는 데 중점을 두었습니다.

본 PoC 코드는 100% Vibe Coding으로 작성되었습니다. 스터디 팀원들의 실습을 위해 작성된 코드로, 이 코드를 기반으로 스터디 팀원들의 PoC 작성에 대한 도움을 제공하고자 합니다.

가장 적합한 플랫폼은 Cursor AI 혹은 VS Code + copilot chat이라고 생각합니다. 구현된 코드의 의미를 생각하며, 직접 구현하거나 혹은 자신만의 프롬프트로 vibe coding을 활용한 PoC 작성을 해 보세요.

`{system_prompt}`: 
```
당신은 transformer architecture를 정확하게 이해하고 있는 한달 1000000 USD를 받는 ML 전문가 및 파이썬 프로그래머입니다.
당신은 사용자의 지시에 맞게 이 이후의 지시에 따라 완벽한 코드를 작성해야합니다.
```

`{user}`:
```
다음 Transformer PoC 구현사항을 적절한 모듈 단위로 분리하여 작성해주세요.

필수 구현 요소
Input Embedding
간단한 사전(token to embedding) 혹은 임의의 입력 벡터 사용
Positional Encoding
Sine/Cosine 방식 또는 간단한 위치 벡터 삽입

Scaled Dot-Product Self-Attention
Query, Key, Value 계산 → 어텐션 score 계산

Multi-Head Attention
여러 개의 Self-Attention을 병렬로 수행하고 통합

Feed Forward Network (FFN)
2층 선형 네트워크 (비선형 활성화 포함)

Residual Connection & Layer Normalization
각 모듈별로 잔차 & 정규화 수행

최소 Encoder Block
위 요소들이 올바른 순서로 연결되었는지 전체 흐름 검증

구현 언어 및 프레임워크
Python, Numpy, PyTorch 사용.
입력/출력
임의의 입력 벡터(예: 랜덤 토큰 시퀀스)를 넣고,
블록의 출력이 잘 나오는지 shape 또는 값으로 검증
```

## 2. 프로젝트 구조

```
transformer_pytorch/
├── embedding.py         # 입력 임베딩 및 위치 인코딩
├── attention.py         # 스케일드 닷-프로덕트 어텐션 및 멀티-헤드 어텐션
├── feed_forward.py      # 피드 포워드 네트워크
├── encoder.py           # 인코더 블록
└── main.py              # 전체 모듈 실행 및 검증
```

## 3. 핵심 모듈 구현

### 3.1 `embedding.py`

-   **`TokenEmbedding`**: 입력으로 들어온 토큰(단어) 시퀀스를 `d_model` 차원의 벡터로 변환합니다. PyTorch의 `nn.Embedding`을 사용합니다.
-   **`PositionalEncoding`**: 사인(Sine)과 코사인(Cosine) 함수를 이용하여 각 토큰의 상대적 또는 절대적 위치 정보를 임베딩 벡터에 추가합니다. 이를 통해 모델이 단어의 순서를 학습할 수 있게 됩니다.

### 3.2 `attention.py`

-   **`ScaledDotProductAttention`**: Query, Key, Value 벡터를 입력받아 어텐션 스코어를 계산하고, 스케일링(`sqrt(d_k)`) 후 소프트맥스(Softmax) 함수를 적용하여 어텐션 가중치를 구합니다. 최종적으로 이 가중치를 Value 벡터에 곱하여 컨텍스트 벡터(Context Vector)를 생성합니다.
-   **`MultiHeadAttention`**: `d_model` 차원의 Query, Key, Value를 여러 개의 헤드(`n_heads`)로 나누어 `ScaledDotProductAttention`을 병렬로 수행합니다. 각 헤드에서 나온 컨텍스트 벡터들을 다시 하나로 합친 후, 선형 레이어를 통과시켜 최종 출력을 생성합니다. 이를 통해 모델이 다양한 관점에서 정보의 연관성을 학습할 수 있습니다.

### 3.3 `feed_forward.py`

-   **`PositionwiseFeedForward`**: 2개의 선형 레이어(Linear Layer)와 비선형 활성화 함수(ReLU)로 구성된 간단한 신경망입니다. 어텐션을 거친 결과에 대해 추가적인 비선형 변환을 수행하여 모델의 표현력을 높입니다.

### 3.4 `encoder.py`

-   **`EncoderBlock`**: Transformer 인코더의 단일 블록을 나타냅니다.
    1.  **첫 번째 서브-레이어**: 멀티-헤드 셀프-어텐션(`MultiHeadAttention`)
    2.  **두 번째 서브-레이어**: 피드 포워드 네트워크(`PositionwiseFeedForward`)
-   각 서브-레이어의 출력에는 **잔차 연결(Residual Connection)** 과 **레이어 정규화(Layer Normalization)** sa가 순차적으로 적용됩니다. 이는 깊은 네트워크에서 발생할 수 있는 기울기 소실(Vanishing Gradient) 문제를 완화하고 학습을 안정화하는 데 도움을 줍니다.

## 4. 실행 및 검증 (`main.py`)

-   **목적**: 위에서 구현한 모듈들이 올바른 순서로 연결되어 정상적으로 동작하는지 검증합니다.
-   **입력 데이터**: 실제 텍스트 데이터 대신, `torch.randint`를 사용하여 임의의 토큰 시퀀스(더미 데이터)를 생성하여 사용합니다.
-   **마스크 (`src_mask`)**: 본 PoC에서는 모든 입력 시퀀스의 길이를 동일하게 고정했기 때문에, 특정 토큰(예: 패딩 토큰)을 계산에서 제외하기 위한 패딩 마스크가 필요하지 않습니다. 따라서 `src_mask`는 `None`으로 설정되었습니다.
-   **검증 방법**: 임의의 입력을 인코더 블록에 통과시킨 후, 최종 출력의 형태(shape)가 기대하는 `(batch_size, seq_len, d_model)`과 일치하는지 확인하고, 실제 출력값을 샘플로 보여줍니다.

## 5. 기술 스택

-   Python
-   PyTorch
-   NumPy (PyTorch 내부 의존성)
