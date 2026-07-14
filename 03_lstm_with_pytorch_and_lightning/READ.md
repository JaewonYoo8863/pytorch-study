# Long Short-Term Memory with PyTorch and Lightning

StatQuest의 **Long Short-Term Memory with PyTorch + Lightning** 강의를 따라가며 LSTM의 내부 구조를 직접 구현하고, PyTorch의 `nn.LSTM`을 활용한 구현과 비교했습니다.

## 학습 목표

이번 실습의 목표는 다음과 같습니다.

1. LSTM의 장기기억과 단기기억이 갱신되는 과정 이해
2. Forget Gate, Input Gate, Output Gate 직접 구현
3. PyTorch의 `nn.LSTM` 사용법 이해
4. Lightning을 이용한 학습 과정 구성
5. 직접 구현한 LSTM과 내장 LSTM 비교

## 파일 구성

```text
03_lstm_with_pytorch_and_lightning/
├── lstm_by_hand.py
├── lightning_lstm.py
└── README.md
```

| 파일 | 설명 |
|---|---|
| `lstm_by_hand.py` | LSTM 내부 연산을 직접 구현한 코드 |
| `lightning_lstm.py` | PyTorch의 `nn.LSTM`을 사용한 코드 |

## 실습 데이터

두 개의 시계열 데이터를 사용합니다.

```python
inputs = torch.tensor([
    [0.0, 0.5, 0.25, 1.0],
    [1.0, 0.5, 0.25, 1.0]
])

labels = torch.tensor([0.0, 1.0])
```

두 입력은 첫 번째 값만 다르고 이후 값은 동일합니다.

```text
[0.0, 0.5, 0.25, 1.0] → 0
[1.0, 0.5, 0.25, 1.0] → 1
```

따라서 모델은 첫 번째 시점의 정보를 마지막 시점까지 기억해야 합니다.

## 1. LSTM 직접 구현

`lstm_by_hand.py`에서는 LSTM 셀의 내부 연산을 직접 구현했습니다.

### LSTM의 기억

LSTM은 두 종류의 기억을 사용합니다.

| 코드 변수 | 일반적인 명칭 | 역할 |
|---|---|---|
| `long_memory` | Cell State | 장기적으로 유지되는 기억 |
| `short_memory` | Hidden State | 현재 시점의 출력 및 단기기억 |

### Forget Gate

Forget Gate는 기존 장기기억을 얼마나 유지할지 결정합니다.

```python
long_remember_percent = torch.sigmoid(
    short_memory * self.wlr1
    + input_value * self.wlr2
    + self.blr1
)
```

시그모이드 함수의 출력 범위는 0부터 1이므로 기존 기억의 유지 비율로 사용할 수 있습니다.

### Input Gate

Input Gate는 새로운 정보를 장기기억에 얼마나 반영할지 결정합니다.

```python
potential_remember_percent = torch.sigmoid(
    short_memory * self.wpr1
    + input_value * self.wpr2
    + self.bpr1
)
```

새롭게 저장할 기억 후보는 `tanh` 함수로 생성합니다.

```python
potential_memory = torch.tanh(
    short_memory * self.wp1
    + input_value * self.wp2
    + self.bp1
)
```

### 장기기억 갱신

기존 기억에서 유지할 부분과 새로운 기억에서 추가할 부분을 더합니다.

```python
updated_long_memory = (
    long_memory * long_remember_percent
    + potential_remember_percent * potential_memory
)
```

수식으로 표현하면 다음과 같습니다.

\[
C_t
=
C_{t-1} \times f_t
+
i_t \times \tilde{C}_t
\]

### Output Gate

Output Gate는 갱신된 장기기억 중 어느 정도를 단기기억으로 출력할지 결정합니다.

```python
output_percent = torch.sigmoid(
    short_memory * self.wo1
    + input_value * self.wo2
    + self.bo1
)

updated_short_memory = (
    torch.tanh(updated_long_memory) * output_percent
)
```

### 순전파

장기기억과 단기기억을 0으로 초기화하고 입력값을 순서대로 처리합니다.

```python
long_memory = 0
short_memory = 0

for input_value in input:
    long_memory, short_memory = self.lstm_unit(
        input_value,
        long_memory,
        short_memory
    )
```

각 시점에서 만들어진 기억은 다음 시점으로 전달됩니다. 모든 입력을 처리한 뒤 마지막 단기기억을 최종 예측값으로 반환합니다.

## 2. PyTorch `nn.LSTM` 사용

`lightning_lstm.py`에서는 LSTM의 내부 게이트를 직접 구현하지 않고 PyTorch의 `nn.LSTM`을 사용했습니다.

```python
self.lstm = nn.LSTM(
    input_size=1,
    hidden_size=1
)
```

### 주요 설정

- `input_size=1`: 각 시점에 입력되는 특성의 개수
- `hidden_size=1`: Hidden State의 크기

현재 데이터는 각 시점마다 숫자 하나만 입력되므로 `input_size`는 1입니다.

### 입력 형태 변환

기존 입력의 형태는 다음과 같습니다.

```text
[4]
```

`nn.LSTM`이 처리할 수 있도록 시퀀스 길이와 특성 수를 분리합니다.

```python
input_transformed = input.view(
    len(input),
    1
)
```

변환 결과는 다음과 같습니다.

```text
[4, 1]
```

- `4`: Sequence Length
- `1`: Input Size

### 마지막 시점의 출력 사용

`nn.LSTM`은 모든 시점의 Hidden State를 반환합니다.

```python
lstm_output, _ = self.lstm(input_transformed)
```

이 실습에서는 전체 시퀀스를 읽은 뒤의 결과가 필요하므로 마지막 출력만 사용합니다.

```python
prediction = lstm_output[-1]
```

## 3. Lightning의 역할

두 모델 모두 `LightningModule`을 상속합니다.

```python
class LSTMbyHand(L.LightningModule):
```

Lightning은 LSTM 구조를 대신 만들어주는 것이 아니라 학습 과정을 관리합니다.

`training_step()`에서는 예측값과 정답을 비교해 손실을 계산합니다.

```python
loss = (output_i - label_i) ** 2
```

`Trainer.fit()`을 실행하면 Lightning이 다음 과정을 자동으로 반복합니다.

```text
데이터 불러오기
→ 순전파
→ 손실 계산
→ 역전파
→ 가중치 업데이트
```

따라서 일반 PyTorch에서 직접 작성하는 다음 코드를 생략할 수 있습니다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## 두 구현 방식 비교

| 구분 | 직접 구현 | `nn.LSTM` 사용 |
|---|---|---|
| LSTM 게이트 | 직접 계산 | PyTorch 내부에서 처리 |
| 가중치 및 편향 | 직접 선언 | 자동 생성 |
| 기억 갱신 | 직접 구현 | 자동 처리 |
| 코드 길이 | 길음 | 짧음 |
| 구조 이해 | 용이함 | 내부 계산이 보이지 않음 |
| 실제 활용 | 학습 목적 | 일반적인 프로젝트 방식 |

직접 구현한 코드는 LSTM의 내부 구조를 이해하는 데 유용합니다. 실제 프로젝트에서는 안정성과 효율성을 위해 대부분 `nn.LSTM`을 사용합니다.

## 실행 방법

필요한 라이브러리를 설치합니다.

```bash
pip install torch lightning
```

직접 구현한 LSTM 실행:

```bash
python lstm_by_hand.py
```

PyTorch의 `nn.LSTM` 버전 실행:

```bash
python lightning_lstm.py
```

## 학습 결과

학습 전에는 가중치가 무작위로 초기화되므로 두 입력에 대한 예측값이 정답과 다르게 나옵니다.

학습이 진행되면서 모델의 출력은 다음 값에 가까워지는 것을 목표로 합니다.

```text
Company A → 0
Company B → 1
```

직접 구현한 모델은 많은 epoch 동안 학습하며, `nn.LSTM`을 사용한 모델은 더 짧고 간단한 코드로 동일한 문제를 학습합니다.

## 정리

이번 실습을 통해 LSTM이 다음 순서로 정보를 처리한다는 것을 확인했습니다.

```text
기존 기억의 유지 비율 결정
→ 새로운 기억 후보 생성
→ 장기기억 갱신
→ 출력할 정보 결정
→ 단기기억 생성
```

LSTM을 직접 구현하면서 각 게이트가 장기기억과 단기기억을 어떻게 제어하는지 이해할 수 있었습니다. 이후 `nn.LSTM`을 사용해 동일한 연산을 훨씬 간단하게 구현했으며, Lightning이 학습 반복, 역전파, 최적화와 로그 기록을 관리한다는 점도 확인했습니다.

## Reference

- StatQuest: Long Short-Term Memory with PyTorch + Lightning
- PyTorch `nn.LSTM`
- Lightning `LightningModule`
