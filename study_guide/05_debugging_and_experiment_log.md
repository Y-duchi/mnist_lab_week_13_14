# Debugging And Experiment Log

이 문서는 구현 중 막혔을 때 확인할 순서와 실험 기록 양식입니다.

## 빠른 디버깅 원칙

1. 에러 메시지의 첫 번째 원인을 봅니다.
2. shape를 먼저 확인합니다.
3. forward 저장값이 backward에서 존재하는지 확인합니다.
4. `params`와 `grads`의 key와 shape를 비교합니다.
5. train mode와 test mode가 섞이지 않았는지 확인합니다.
6. 작은 배열로 손계산 가능한 예제를 먼저 통과시킵니다.

## 증상별 점검

### `NotImplementedError`가 난다

확인:

- 해당 TODO 함수가 아직 남아 있습니다.
- 테스트가 어느 파일을 import하는지 확인합니다.

명령:

```bash
rg -n "TODO|NotImplementedError|pass" src tests
```

### shape mismatch

확인:

- Affine의 `W` shape가 `(input_dim, output_dim)`인가?
- `x @ W + b`에서 `b`가 `(output_dim,)`인가?
- `dW = x.T @ dout`인가?
- `db = sum(dout, axis=0)`인가?
- BatchNorm의 `gamma`, `beta`가 feature dimension과 같은가?

디버그 출력 예:

```python
print("x", x.shape)
print("W", W.shape)
print("b", b.shape)
print("out", out.shape)
print("dout", dout.shape)
```

### loss가 NaN이 된다

확인:

- Softmax에서 max-shift를 했는가?
- Cross Entropy에서 `np.clip`을 했는가?
- learning rate가 너무 크지 않은가?
- BatchNorm에서 `sqrt(var + eps)`를 쓰는가?

먼저 시도:

- lr `0.001`에서 시작
- NaN이 나면 `0.0005`, `0.0001`로 낮춤
- Softmax 출력의 min/max 확인

### loss가 줄지 않는다

확인:

- `optimizer.update(model.params, model.grads)`가 호출되는가?
- `params`가 실제로 바뀌는가?
- `dout[np.arange(batch_size), y_batch] -= 1` 뒤에 `dout /= batch_size`를 했는가?
- ReLU backward에서 mask 방향이 반대로 되지 않았는가?
- Dropout ratio가 너무 크지 않은가?

간단 확인:

```python
before = model.params["W1"].copy()
optimizer.update(model.params, model.grads)
print(np.mean(np.abs(model.params["W1"] - before)))
```

### train 정확도는 오르는데 test 정확도가 낮다

가능성:

- 과적합
- Dropout/BatchNorm test mode 오류
- 데이터 shuffle 또는 batch 처리 오류
- learning rate가 부적절함

확인:

- `model.predict`가 `train=False`를 쓰는가?
- BatchNorm test mode가 running mean/var를 쓰는가?
- Dropout test mode가 random mask를 만들지 않는가?

### test mode 결과가 실행마다 달라진다

가능성:

- Dropout이 test mode에서도 random mask를 만들고 있습니다.
- BatchNorm이 test mode에서도 batch statistics를 사용하고 있습니다.

확인:

```python
y1 = model.predict(x_test[:100])
y2 = model.predict(x_test[:100])
print(np.max(np.abs(y1 - y2)))
```

동일한 model과 동일한 input이면 test mode 결과는 같아야 합니다.

## 전체 구현 점검 코드 조각

이 조각은 구현 후 임시로 노트북이나 별도 셀에서 확인합니다.

```python
for key in model.params:
    print(key, model.params[key].shape, model.grads[key].shape)
```

기대:

```text
모든 key에서 params[key].shape == grads[key].shape
```

Softmax 확인:

```python
y_pred = model.forward(x_train[:5], train=True)
print(y_pred.shape)
print(y_pred.sum(axis=1))
print(y_pred.min(), y_pred.max())
```

기대:

```text
(5, 10)
각 행 합이 1에 가까움
min >= 0, max <= 1
```

## 실험 기록 양식

아래 표를 REPORT 작성 전까지 계속 채웁니다.

| 실험 ID | 구조 | BatchNorm | Dropout | Optimizer | LR | Epochs | Batch | Init | Train Loss 마지막 | Test Acc | 시간 | 메모 |
|---|---|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|
| exp-001 | 784-512-256-10 | yes | 0.5 | Adam | 0.001 | 5 | 128 | He |  |  |  | baseline |
| exp-002 | 784-512-256-10 | yes | 0.3 | Adam | 0.001 | 10 | 128 | He |  |  |  | dropout 조정 |
| exp-003 | 784-512-256-10 | yes | 0.2 | Adam | 0.0005 | 15 | 128 | He |  |  |  | lr 조정 |

## REPORT에 쓸 문장 재료

### 실험 목적

```text
본 실험은 딥러닝 프레임워크 없이 NumPy만 사용하여 MNIST 손글씨 숫자 분류용 다층 퍼셉트론을 구현하고, forward/loss/backward/update 학습 흐름을 직접 확인하는 것을 목표로 한다.
```

### 모델 구조

```text
입력은 28x28 이미지를 펼친 784차원 벡터이며, 은닉층은 Affine, BatchNorm, ReLU, Dropout 순서로 구성했다. 출력층은 10개 클래스 logit을 만든 뒤 Softmax로 확률을 계산했다.
```

### 회고 관점

- loss가 안정적으로 감소했는가?
- BatchNorm 사용 여부가 수렴 속도에 영향을 주었는가?
- Dropout ratio를 바꾸었을 때 test accuracy가 어떻게 변했는가?
- learning rate가 너무 크거나 작을 때 어떤 현상이 있었는가?
- 구현 중 가장 많이 발생한 shape 오류는 무엇이었는가?

