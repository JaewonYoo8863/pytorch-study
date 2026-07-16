# Decoder-Only Transformer with PyTorch and Lightning

StatQuest의 **Coding a ChatGPT Like Transformer From Scratch in PyTorch** 강의를 바탕으로 Decoder-only Transformer를 직접 구현한 실습입니다.

토큰 임베딩, 위치 인코딩, Masked Self-Attention, Residual Connection과 다음 토큰 생성 과정을 PyTorch 코드로 구현하고, PyTorch Lightning을 이용해 모델을 학습합니다.

---

## 학습 목표

* Decoder-only Transformer의 전체 데이터 흐름을 이해한다.
* 토큰과 정수 ID를 서로 변환하는 vocabulary를 구성한다.
* 다음 토큰 예측을 위한 입력과 라벨을 만든다.
* Teacher forcing 방식의 학습 데이터를 이해한다.
* Sin과 Cos를 이용한 Position Encoding을 구현한다.
* Query, Key, Value를 이용한 Self-Attention을 구현한다.
* Causal Mask가 미래 토큰을 차단하는 원리를 이해한다.
* Residual Connection을 적용한다.
* PyTorch Lightning으로 Transformer를 학습한다.
* 자기회귀 방식으로 다음 토큰을 생성한다.

---

## 파일 구성

```text
05_decoder_only_transformer/
├─ decoder_only_transformer.py
└─ README.md
```

---

## 학습 데이터

모델은 다음 두 질문에 같은 답을 생성하도록 학습합니다.

```text
what is statquest <EOS> → awesome <EOS>
statquest is what <EOS> → awesome <EOS>
```

Vocabulary는 다음 다섯 개 토큰으로 구성합니다.

```python
token_to_id = {
    "what": 0,
    "is": 1,
    "statquest": 2,
    "awesome": 3,
    "<EOS>": 4
}
```

`token_to_id`는 문자열을 모델 입력용 정수로 바꾸고, `id_to_token`은 모델이 예측한 정수를 다시 문자열로 변환합니다.

---

## 입력과 라벨

Decoder-only 언어 모델은 현재까지의 토큰을 보고 다음 토큰을 예측합니다.

첫 번째 문장의 입력과 라벨은 다음과 같습니다.

```text
입력: what → is → statquest → <EOS> → awesome
정답: is → statquest → <EOS> → awesome → <EOS>
```

라벨은 입력을 한 칸 왼쪽으로 이동한 형태입니다.

학습 입력에 정답 토큰인 `awesome`까지 포함하여 다음 토큰을 학습시키는 방식을 Teacher Forcing이라고 합니다.

---

## 모델 구조

```text
Token ID
   ↓
Word Embedding
   ↓
Position Encoding
   ↓
Masked Self-Attention
   ↓
Residual Connection
   ↓
Fully Connected Layer
   ↓
Next-token logits
```

이 실습에서는 다음 설정을 사용합니다.

```text
Vocabulary size = 5
Embedding dimension = 2
Maximum sequence length = 6
Attention head = 1
Decoder layer = 1
```

---

## Position Encoding

Self-Attention은 토큰을 동시에 처리하기 때문에 토큰 순서를 자체적으로 구분하지 못합니다.

각 위치에 서로 다른 Sin과 Cos 값을 만들어 Word Embedding에 더합니다.

```python
pe[:, 0::2] = torch.sin(position * div_term)
pe[:, 1::2] = torch.cos(position * div_term)
```

위치 인코딩은 학습되는 값이 아니지만 모델과 함께 저장되고 CPU와 GPU 사이를 같이 이동해야 합니다.

```python
self.register_buffer("pe", pe)
```

`forward()`에서는 입력 길이에 필요한 위치 인코딩만 선택합니다.

```python
return word_embeddings + self.pe[:word_embeddings.size(0), :]
```

---

## Self-Attention

입력 벡터를 세 개의 Linear Layer에 통과시켜 Query, Key, Value를 만듭니다.

```python
self.W_q = nn.Linear(d_model, d_model, bias=False)
self.W_k = nn.Linear(d_model, d_model, bias=False)
self.W_v = nn.Linear(d_model, d_model, bias=False)
```

각 토큰 사이의 유사도는 Query와 전치한 Key의 행렬곱으로 계산합니다.

```python
sims = torch.matmul(
    q,
    k.transpose(0, 1)
)
```

유사도 값이 지나치게 커지는 것을 막기 위해 `sqrt(d_model)`로 나눕니다.

```python
scaled_sims = sims / (d_model ** 0.5)
```

Softmax를 적용한 Attention 비율과 Value를 곱해 최종 Attention 출력을 계산합니다.

```python
attention_percents = F.softmax(
    scaled_sims,
    dim=1
)

attention_scores = torch.matmul(
    attention_percents,
    v
)
```

---

## Causal Mask

Decoder-only Transformer는 현재 위치 이후의 미래 토큰을 볼 수 없어야 합니다.

아래 삼각 행렬을 이용해 현재 및 이전 위치만 참고할 수 있도록 합니다.

```python
mask = torch.tril(
    torch.ones(sequence_length, sequence_length)
)

mask = mask == 0
```

미래 위치의 Attention score를 매우 큰 음수로 변경합니다.

```python
scaled_sims = scaled_sims.masked_fill(
    mask=mask,
    value=-1e9
)
```

Softmax를 적용하면 마스킹된 위치의 Attention 비율이 거의 0이 됩니다.

---

## Residual Connection

Position Encoding이 적용된 입력과 Self-Attention 결과를 더합니다.

```python
residual_connection_values = (
    position_encoded
    + self_attention_values
)
```

이를 통해 Attention 처리 이전의 토큰 정보를 출력에 함께 전달합니다.

---

## Output Layer

각 토큰의 Attention 출력은 2차원 벡터입니다.

마지막 Linear Layer는 이를 Vocabulary 크기인 5차원 logits로 변환합니다.

```python
self.fc_layer = nn.Linear(
    in_features=2,
    out_features=5
)
```

출력 shape은 다음과 같습니다.

```text
[sequence length, vocabulary size]
```

각 위치에는 다음 토큰이 `what`, `is`, `statquest`, `awesome`, `<EOS>`일 점수가 들어 있습니다.

---

## Loss

다음 토큰 예측은 Vocabulary에 포함된 토큰 중 하나를 선택하는 다중 분류 문제입니다.

```python
self.loss = nn.CrossEntropyLoss()
```

`CrossEntropyLoss`는 raw logits를 입력으로 받으므로 모델의 `forward()`에서는 Softmax를 적용하지 않습니다.

```python
output = self.forward(input_tokens[0])
loss = self.loss(output, labels[0])
```

---

## PyTorch Lightning

최종 모델은 `L.LightningModule`을 상속합니다.

### `forward()`

```text
Embedding
→ Position Encoding
→ Causal Mask 생성
→ Masked Self-Attention
→ Residual Connection
→ Output Linear Layer
```

### `training_step()`

모델 예측값과 정답 토큰을 이용해 Cross Entropy Loss를 계산합니다.

### `configure_optimizers()`

Adam 옵티마이저와 학습률을 설정합니다.

```python
return Adam(
    self.parameters(),
    lr=0.1
)
```

모델은 30 epochs 동안 학습합니다.

```python
trainer = L.Trainer(max_epochs=30)

trainer.fit(
    model,
    train_dataloaders=dataloader
)
```

---

## Autoregressive Generation

생성 과정에서는 현재까지 생성한 모든 토큰을 모델에 입력하고, 마지막 위치의 예측만 사용합니다.

```python
predictions = model(model_input)

predicted_id = torch.tensor([
    torch.argmax(predictions[-1, :])
])
```

예측한 토큰이 `<EOS>`가 아니면 기존 입력 뒤에 추가합니다.

```python
model_input = torch.cat(
    (model_input, predicted_id)
)
```

이 과정을 `<EOS>`가 나오거나 최대 길이에 도달할 때까지 반복합니다.

```text
질문 입력
→ 다음 토큰 예측
→ 입력 뒤에 예측 토큰 추가
→ 전체 시퀀스를 다시 입력
→ 다음 토큰 예측
```

학습이 완료되면 두 질문 모두 다음 결과를 생성합니다.

```text
awesome
<EOS>
```

---

## Tensor Shape

학습 시 sequence length는 5이고 `d_model`은 2입니다.

```text
Token IDs             [5]
Word Embeddings       [5, 2]
Position Encoded      [5, 2]
Q, K, V               [5, 2]
Attention Similarity  [5, 5]
Attention Output      [5, 2]
Output Logits         [5, 5]
Labels                [5]
```

첫 번째 `5`는 sequence length이고, 마지막 출력의 두 번째 `5`는 vocabulary size입니다.

---

## 실행 방법

필요한 라이브러리를 설치합니다.

```bash
pip install torch lightning
```

코드를 실행합니다.

```bash
python decoder_only_transformer.py
```

---

## 구현 범위

이 코드는 Decoder-only Transformer의 핵심 원리를 확인하기 위한 교육용 구현입니다.

포함된 구성은 다음과 같습니다.

```text
Word Embedding
Sinusoidal Position Encoding
Single-Head Masked Self-Attention
Residual Connection
Output Linear Layer
Autoregressive Generation
```

다음 구성은 생략되어 있습니다.

```text
Multi-Head Attention
Feed Forward Network
Layer Normalization
Dropout
여러 개의 Decoder Block
Tokenizer
대규모 학습 데이터
```

따라서 실제 ChatGPT 전체 모델보다는 GPT 계열 모델의 핵심 작동 원리를 보여주는 최소 구조에 가깝습니다.

---

## 핵심 정리

* Decoder-only Transformer는 현재까지의 토큰을 이용해 다음 토큰을 예측한다.
* 학습 라벨은 입력 토큰을 한 칸 이동한 형태로 구성한다.
* Teacher forcing은 학습 시 이전 정답 토큰을 입력으로 제공한다.
* Word Embedding은 토큰 ID를 벡터로 변환한다.
* Position Encoding은 토큰 벡터에 순서 정보를 추가한다.
* Q와 K의 행렬곱은 토큰 사이의 유사도를 계산한다.
* Causal Mask는 미래 토큰의 Attention 비율을 0으로 만든다.
* Attention 비율과 V의 행렬곱으로 문맥이 반영된 벡터를 만든다.
* Residual Connection은 Attention 입력을 출력에 다시 더한다.
* 마지막 Linear Layer는 각 토큰에 대한 logits를 출력한다.
* 생성 시 마지막 위치의 logits에서 다음 토큰을 선택한다.
* 예측한 토큰을 입력 뒤에 붙이는 과정을 반복하면 문장이 생성된다.

---

## Reference

* StatQuest with Josh Starmer
* Coding a ChatGPT Like Transformer From Scratch in PyTorch
* Decoder-Only Transformer with PyTorch and Lightning
