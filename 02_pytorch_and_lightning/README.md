# 02. Neural Networks with PyTorch and Lightning

StatQuest의 **Introduction to Coding Neural Networks with PyTorch and Lightning** 강의를 따라가며 실습한 내용을 정리한 예제입니다.

이 장에서는 이전에 순수 PyTorch로 구현했던 간단한 신경망을 PyTorch Lightning 방식으로 다시 작성하고, 마지막 편향값인 `final_bias`를 학습합니다.

---

## 1. 학습 목표

- `LightningModule`을 이용한 신경망 정의
- `forward()`를 통한 순전파 구현
- `TensorDataset`과 `DataLoader`를 이용한 데이터 구성
- `training_step()`을 이용한 손실 함수 계산
- `configure_optimizers()`를 이용한 Optimizer 설정
- Lightning `Trainer`를 이용한 모델 학습
- Learning Rate Finder를 이용한 학습률 탐색
- 학습 전후 출력 결과 시각화

---

## 2. PyTorch Lightning

PyTorch Lightning은 PyTorch를 대체하는 라이브러리가 아니라, PyTorch 위에서 학습 코드를 구조화하고 반복 작업을 줄여주는 프레임워크입니다.

순수 PyTorch에서는 아래와 같은 학습 과정을 직접 작성해야 합니다.

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
````

Lightning에서는 `training_step()`에서 손실값을 반환하고, `configure_optimizers()`에서 Optimizer를 설정하면 `Trainer`가 역전파와 파라미터 갱신을 자동으로 처리합니다.

---

## 3. 신경망 구조

이 예제의 신경망은 입력값인 `Dose`를 받아 두 개의 ReLU 노드를 통과한 뒤 최종 출력값인 `Effectiveness`를 계산합니다.

전체 계산 과정은 다음과 같습니다.

```text
Input
 ├─ Top ReLU
 └─ Bottom ReLU
        ↓
   Weighted Sum
        ↓
    Final Bias
        ↓
    Final ReLU
        ↓
      Output
```

순전파는 `forward()` 메서드에서 정의합니다.

```python
def forward(self, input):
    input_to_top_relu = input * self.w00 + self.b00
    top_relu_output = F.relu(input_to_top_relu)
    scaled_top_relu_output = top_relu_output * self.w01

    input_to_bottom_relu = input * self.w10 + self.b10
    bottom_relu_output = F.relu(input_to_bottom_relu)
    scaled_bottom_relu_output = bottom_relu_output * self.w11

    input_to_final_relu = (
        scaled_top_relu_output
        + scaled_bottom_relu_output
        + self.final_bias
    )

    output = F.relu(input_to_final_relu)

    return output
```

---

## 4. 고정된 모델

`BasicLightning` 클래스는 모든 가중치와 편향값이 고정된 모델입니다.

```python
class BasicLightning(L.LightningModule):
```

각 파라미터의 `requires_grad` 값은 `False`이므로 학습 과정에서 값이 변경되지 않습니다.

```python
self.w00 = nn.Parameter(
    torch.tensor(1.7),
    requires_grad=False
)
```

최종 편향값도 처음부터 `-16.0`으로 설정되어 있습니다.

```python
self.final_bias = nn.Parameter(
    torch.tensor(-16.0),
    requires_grad=False
)
```

---

## 5. 학습 가능한 모델

`BasicLightningTrain` 클래스에서는 나머지 가중치와 편향은 고정하고, `final_bias`만 학습합니다.

```python
self.final_bias = nn.Parameter(
    torch.tensor(0.0),
    requires_grad=True
)
```

처음에는 `final_bias`가 `0.0`이기 때문에 모델의 출력이 정답 데이터와 크게 다릅니다.

학습을 통해 `final_bias`를 적절한 값으로 조정합니다.

---

## 6. 학습 데이터

학습에 사용하는 입력값은 다음과 같습니다.

```python
inputs = torch.tensor(
    [0.0, 0.5, 1.0] * 100
)
```

각 입력값에 대한 정답은 다음과 같습니다.

```python
labels = torch.tensor(
    [0.0, 1.0, 0.0] * 100
)
```

입력값과 정답값을 `TensorDataset`으로 묶습니다.

```python
dataset = TensorDataset(
    inputs,
    labels
)
```

이후 `DataLoader`를 이용해 데이터를 모델에 전달합니다.

```python
dataloader = DataLoader(dataset)
```

`DataLoader`를 사용하면 데이터를 배치 단위로 불러오거나, 데이터를 섞거나, 큰 데이터셋을 효율적으로 처리할 수 있습니다.

---

## 7. training_step

`training_step()`은 하나의 배치에 대해 모델의 출력값과 손실값을 계산합니다.

```python
def training_step(self, batch, batch_idx):
    input_i, label_i = batch

    output_i = self.forward(input_i)

    loss = (output_i - label_i) ** 2

    return loss
```

이 예제에서는 예측값과 정답값 차이의 제곱을 손실값으로 사용합니다.

```python
loss = (output_i - label_i) ** 2
```

Lightning에서는 계산된 `loss`를 반환하면 `Trainer`가 역전파와 파라미터 업데이트를 자동으로 수행합니다.

---

## 8. configure_optimizers

`configure_optimizers()`에서는 모델 학습에 사용할 Optimizer를 설정합니다.

```python
def configure_optimizers(self):
    return SGD(
        self.parameters(),
        lr=self.learning_rate
    )
```

이 예제에서는 확률적 경사하강법인 `SGD`를 사용합니다.

`requires_grad=True`로 설정된 파라미터는 `final_bias`뿐이므로 실제로는 `final_bias`만 업데이트됩니다.

---

## 9. Learning Rate Finder

처음에는 임시 학습률을 설정합니다.

```python
self.learning_rate = 0.01
```

이후 Lightning의 `Tuner`와 `lr_find()`를 이용해 적절한 학습률을 탐색합니다.

```python
trainer = L.Trainer(max_epochs=20)

tuner = L.pytorch.tuner.Tuner(trainer)

lr_find_results = tuner.lr_find(
    model,
    train_dataloaders=dataloader,
    min_lr=0.001,
    max_lr=1.0,
    early_stop_threshold=None
)
```

추천된 학습률은 다음과 같이 가져옵니다.

```python
new_lr = lr_find_results.suggestion()
```

추천 학습률을 모델에 적용합니다.

```python
model.learning_rate = new_lr
```

---

## 10. 모델 학습

Lightning에서는 `Trainer.fit()`을 이용해 학습을 실행합니다.

```python
trainer.fit(
    model,
    train_dataloaders=dataloader
)
```

`Trainer`는 내부적으로 다음 과정을 반복합니다.

1. `DataLoader`에서 데이터 가져오기
2. `training_step()` 실행
3. 손실값 계산
4. Gradient 초기화
5. 역전파 실행
6. Optimizer를 이용한 파라미터 갱신

순수 PyTorch에서 직접 작성했던 학습 반복문을 Lightning이 대신 관리합니다.

---

## 11. 학습 결과

학습 전에는 `final_bias`가 `0.0`으로 설정되어 있어 출력값이 정답과 크게 다릅니다.

학습이 끝나면 `final_bias`가 정답 데이터에 맞는 방향으로 최적화됩니다.

```python
print(model.final_bias.data)
```

학습된 모델을 이용해 다시 출력값을 계산합니다.

```python
output_values = model(input_doses)
```

최종 그래프에서는 입력값 `0.5` 부근에서 출력값이 높아지고, 입력값 `0.0`과 `1.0` 부근에서는 출력값이 낮아지는 형태를 확인할 수 있습니다.

---

## 12. PyTorch와 Lightning 비교

### 순수 PyTorch

```python
for epoch in range(100):
    for input_i, label_i in data:
        optimizer.zero_grad()

        output_i = model(input_i)
        loss = (output_i - label_i) ** 2

        loss.backward()
        optimizer.step()
```

### PyTorch Lightning

```python
def training_step(self, batch, batch_idx):
    input_i, label_i = batch
    output_i = self(input_i)
    loss = (output_i - label_i) ** 2
    return loss
```

```python
trainer.fit(
    model,
    train_dataloaders=dataloader
)
```

Lightning에서는 학습 반복문을 직접 작성하지 않고, 손실 함수와 Optimizer 설정에 집중할 수 있습니다.

---

## 13. 실행 방법

필요한 라이브러리를 설치합니다.

```bash
pip install torch lightning matplotlib seaborn
```

Python 파일을 실행합니다.

```bash
python basic_lightning.py
```

---

## 14. 핵심 정리

* `LightningModule`은 PyTorch의 `nn.Module`을 기반으로 합니다.
* 신경망의 순전파는 `forward()`에 정의합니다.
* 손실 계산은 `training_step()`에 정의합니다.
* Optimizer는 `configure_optimizers()`에서 설정합니다.
* `Trainer`가 학습 반복, 역전파, 파라미터 업데이트를 관리합니다.
* `lr_find()`를 이용해 적절한 학습률을 탐색할 수 있습니다.
* 이 예제에서는 다른 파라미터는 고정하고 `final_bias`만 학습합니다.

---

## Reference

* StatQuest: Introduction to Coding Neural Networks with PyTorch and Lightning

```
```
