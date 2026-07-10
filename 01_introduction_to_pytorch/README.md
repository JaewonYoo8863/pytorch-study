# 01. Introduction to PyTorch

## 배운 내용

- `nn.Module`을 이용한 모델 정의
- `nn.Parameter`와 `requires_grad`
- `forward()`를 통한 순전파
- `loss.backward()`를 통한 역전파
- SGD를 이용한 `final_bias` 최적화

## 핵심 코드

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
