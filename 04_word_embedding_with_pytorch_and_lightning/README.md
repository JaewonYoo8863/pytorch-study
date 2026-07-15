# Word Embedding with PyTorch and Lightning

StatQuest의 **Word Embedding in PyTorch + Lightning** 강의를 바탕으로 단어 임베딩 네트워크를 구현한 실습입니다.

같은 구조의 임베딩 네트워크를 다음 세 가지 방식으로 구현하며, PyTorch에서 추상화 수준이 높아질수록 코드가 어떻게 간결해지는지 비교합니다.

1. `nn.Parameter`를 사용한 직접 구현
2. `nn.Linear`를 사용한 행렬 연산 구현
3. 학습된 가중치를 `nn.Embedding.from_pretrained()`로 불러오기

---

## 학습 목표

* 원-핫 인코딩을 이용해 토큰을 신경망 입력으로 표현한다.
* 입력층과 은닉층 사이의 가중치가 단어 임베딩이 되는 과정을 이해한다.
* `nn.Parameter`와 `nn.Linear`의 관계를 이해한다.
* 원-핫 벡터와 가중치 행렬의 곱이 임베딩 조회와 같은 이유를 이해한다.
* 학습된 가중치를 `nn.Embedding` 객체로 변환한다.
* PyTorch Lightning을 이용해 학습 반복문을 간결하게 작성한다.

---

## 파일 구성

```text
04_word_embedding_with_pytorch_and_lightning/
├─ 01_word_embedding_from_scratch.py
├─ 02_word_embedding_with_linear.py
├─ 03_load_pretrained_embedding.py
└─ README.md
```

### `01_word_embedding_from_scratch.py`

입력층과 은닉층, 은닉층과 출력층 사이의 가중치를 각각 `nn.Parameter`로 선언합니다.

행렬 연산을 사용하지 않고 각 입력값과 가중치의 곱을 직접 더하면서 신경망 내부의 계산 과정을 확인합니다.

### `02_word_embedding_with_linear.py`

첫 번째 코드에서 직접 작성했던 가중치 계산을 두 개의 `nn.Linear` 계층으로 대체합니다.

```python
nn.Linear(in_features=4, out_features=2, bias=False)
nn.Linear(in_features=2, out_features=4, bias=False)
```

첫 번째 Linear 계층의 가중치가 각 토큰의 2차원 임베딩 값에 해당합니다.

### `03_load_pretrained_embedding.py`

두 번째 코드에서 학습한 입력층 가중치를 `nn.Embedding.from_pretrained()`에 전달합니다.

`nn.Linear`의 가중치와 `nn.Embedding`의 가중치 배치 방향이 다르기 때문에 전치 행렬을 사용합니다.

```python
word_embeddings = nn.Embedding.from_pretrained(
    trained_weights.T
)
```

이후 원-핫 벡터 대신 토큰의 정수 인덱스로 임베딩을 조회할 수 있습니다.

---

## 실습 데이터

실습에서 사용하는 토큰은 다음 네 개입니다.

```text
Troll2
is
great
Gymkata
```

각 토큰은 처음에 원-핫 벡터로 표현합니다.

```text
Troll2  → [1, 0, 0, 0]
is      → [0, 1, 0, 0]
great   → [0, 0, 1, 0]
Gymkata → [0, 0, 0, 1]
```

모델은 현재 토큰을 입력받아 다음에 등장할 토큰을 예측하도록 학습합니다.

```text
Troll2  → is
is      → great
great   → Gymkata
Gymkata → is
```

`Troll2`와 `Gymkata`는 모두 다음 단어로 `is`가 등장하기 때문에 비슷한 문맥을 가진 토큰으로 학습됩니다.

---

## 모델 구조

전체 네트워크 구조는 다음과 같습니다.

```text
입력 노드 4개
      ↓
임베딩 차원 2개
      ↓
출력 노드 4개
```

입력층과 은닉층 사이에는 편향과 활성화 함수를 사용하지 않습니다.

따라서 원-핫 입력 벡터를 넣으면 입력층 가중치 중 해당 토큰에 대응하는 값이 그대로 은닉층에 전달됩니다. 이 은닉층의 두 값이 해당 토큰의 2차원 임베딩입니다.

---

## 1. `nn.Parameter`를 이용한 직접 구현

첫 번째 구현에서는 모든 가중치를 개별적인 `nn.Parameter`로 선언합니다.

입력 노드 4개와 은닉 노드 2개 사이에는 총 8개의 가중치가 필요하고, 은닉 노드 2개와 출력 노드 4개 사이에도 총 8개의 가중치가 필요합니다.

순전파 과정에서도 입력과 가중치의 곱을 각각 직접 계산합니다.

이 방식은 코드가 길지만, 신경망에서 실제로 수행되는 행렬곱을 개별 스칼라 연산 수준에서 확인할 수 있다는 장점이 있습니다.

---

## 2. `nn.Linear`를 이용한 구현

두 번째 구현에서는 직접 선언했던 여러 개의 파라미터를 `nn.Linear` 두 개로 대체합니다.

```python
self.input_to_hidden = nn.Linear(
    in_features=4,
    out_features=2,
    bias=False
)

self.hidden_to_output = nn.Linear(
    in_features=2,
    out_features=4,
    bias=False
)
```

순전파도 다음과 같이 간단해집니다.

```python
hidden = self.input_to_hidden(input_tensor)
output = self.hidden_to_output(hidden)
```

`nn.Linear`는 내부적으로 가중치 행렬을 생성하고, 해당 가중치를 학습 가능한 파라미터로 자동 등록합니다.

따라서 첫 번째 방식과 계산 원리는 같지만 파라미터 선언과 행렬 연산을 PyTorch가 대신 관리합니다.

---

## 3. `nn.Embedding.from_pretrained()` 사용

세 번째 단계에서는 새로운 임베딩 모델을 다시 학습하지 않습니다.

두 번째 모델의 입력층에서 학습된 가중치를 가져와 `nn.Embedding` 객체로 변환합니다.

원-핫 벡터와 임베딩 가중치 행렬을 곱하면 결국 원-핫 벡터에서 값이 1인 위치에 해당하는 가중치만 선택됩니다.

따라서 다음 두 연산은 같은 결과를 만듭니다.

```text
원-핫 벡터 × 가중치 행렬
```

```text
토큰 인덱스로 임베딩 행 조회
```

`nn.Embedding`은 불필요한 원-핫 벡터와 행렬곱을 만들지 않고, 토큰 인덱스에 해당하는 임베딩 값을 바로 조회합니다.

---

## 세 가지 방식 비교

| 방식             | 입력 형태  | 가중치 관리              | 특징                        |
| -------------- | ------ | ------------------- | ------------------------- |
| `nn.Parameter` | 원-핫 벡터 | 가중치를 하나씩 직접 선언      | 내부 계산 과정을 명확하게 확인할 수 있음   |
| `nn.Linear`    | 원-핫 벡터 | 가중치 행렬을 PyTorch가 관리 | 같은 계산을 훨씬 짧은 코드로 구현       |
| `nn.Embedding` | 토큰 인덱스 | 임베딩 행렬을 조회 테이블로 관리  | 원-핫 행렬곱 없이 필요한 임베딩을 바로 조회 |

세 방식의 핵심 계산은 동일하지만 코드의 추상화 수준과 입력 방식이 다릅니다.

---

## Loss와 Softmax

모델의 `forward()`에서는 Softmax를 적용하지 않고 각 클래스에 대한 raw score인 logit을 반환합니다.

```python
self.loss = nn.CrossEntropyLoss()
```

`nn.CrossEntropyLoss`가 내부적으로 LogSoftmax와 Negative Log Likelihood Loss를 함께 처리하기 때문입니다.

따라서 학습 중에는 모델 출력에 Softmax를 별도로 적용하지 않습니다.

학습이 끝난 후 예측 결과를 확률로 확인할 때만 Softmax를 사용합니다.

---

## PyTorch Lightning의 역할

모델은 `L.LightningModule`을 상속하고 다음 메서드를 구현합니다.

### `forward()`

입력을 모델에 통과시켜 예측 결과를 반환합니다.

### `training_step()`

배치에서 입력과 정답을 꺼내 모델의 예측값과 loss를 계산합니다.

### `configure_optimizers()`

Adam 옵티마이저와 학습률을 설정합니다.

```python
Adam(self.parameters(), lr=0.1)
```

실제 학습은 다음 코드로 실행합니다.

```python
trainer = L.Trainer(max_epochs=100)
trainer.fit(model, train_dataloaders=dataloader)
```

Lightning이 다음 과정을 자동으로 수행합니다.

```text
zero_grad()
forward()
loss 계산
backward()
optimizer.step()
```

---

## 학습 결과

학습 전과 학습 후의 입력층 가중치를 2차원 산점도로 비교합니다.

```text
x축: 첫 번째 임베딩 값
y축: 두 번째 임베딩 값
```

학습 후에는 비슷한 문맥에서 사용된 `Troll2`와 `Gymkata`의 임베딩이 가까운 위치로 이동합니다.

이를 통해 단어 임베딩이 단순히 단어를 숫자로 변환하는 것이 아니라, 학습 데이터의 문맥 관계를 임베딩 공간에 반영한다는 것을 확인할 수 있습니다.

---

## 실행 방법

필요한 라이브러리를 설치합니다.

```bash
pip install torch lightning pandas matplotlib seaborn
```

각 코드를 순서대로 실행합니다.

```bash
python 01_word_embedding_from_scratch.py
python 02_word_embedding_with_linear.py
python 03_load_pretrained_embedding.py
```

세 번째 코드는 두 번째 코드에서 학습한 입력층 가중치를 사용하는 단계이므로, 코드상으로 학습된 가중치가 전달되어야 합니다.

---

## 핵심 정리

* 단어 임베딩은 각 토큰을 저차원 실수 벡터로 표현한다.
* 입력층과 은닉층 사이의 가중치가 토큰의 임베딩 값이 된다.
* `nn.Linear`는 직접 작성한 가중치 행렬과 행렬곱을 하나의 계층으로 캡슐화한다.
* 원-핫 벡터와 가중치 행렬의 곱은 해당 토큰의 임베딩 행을 선택하는 것과 같다.
* `nn.Embedding`은 원-핫 행렬곱 대신 토큰 인덱스로 임베딩을 바로 조회한다.
* `nn.Embedding.from_pretrained()`를 이용하면 이미 학습된 임베딩 가중치를 재사용할 수 있다.
* 비슷한 문맥에서 등장하는 토큰은 학습 후 임베딩 공간에서도 가까워진다.

---

## Reference

* StatQuest with Josh Starmer
* Word Embedding in PyTorch + Lightning
