# Math And NumPy Reference

이 문서는 구현 중 자주 확인할 수식과 shape 기준입니다. 완성 코드가 아니라 검산용 기준입니다.

## 공통 기호

- `N`: batch size
- `D`: input dimension
- `H`: hidden dimension
- `C`: class count, MNIST에서는 10
- `x`: 입력
- `W`: weight
- `b`: bias
- `dout`: 다음 layer에서 넘어온 gradient
- `dx`: 현재 layer 입력에 대한 gradient

## Affine

Forward:

```text
out = x @ W + b
```

Shape:

```text
x:   (N, D)
W:   (D, H)
b:   (H,)
out: (N, H)
```

Backward:

```text
dW = x.T @ dout
db = sum(dout, axis=0)
dx = dout @ W.T
```

Shape:

```text
dout: (N, H)
dW:   (D, H)
db:   (H,)
dx:   (N, D)
```

주의:

- forward 때 `x`를 저장해야 backward에서 `dW`를 계산할 수 있습니다.
- `db`는 batch 방향인 axis 0으로 더합니다.

## ReLU

Forward:

```text
out = max(0, x)
mask = x > 0
```

Backward:

```text
dx = dout * mask
```

과제 테스트 기준:

- `x <= 0` 위치의 gradient는 0입니다.
- `x > 0` 위치의 gradient만 통과합니다.

## Softmax

Forward:

```text
shifted = x - max(x, axis=1, keepdims=True)
exp_x = exp(shifted)
out = exp_x / sum(exp_x, axis=1, keepdims=True)
```

Shape:

```text
x:   (N, C)
out: (N, C)
```

수치 안정성:

- `exp` 전에 각 행의 최댓값을 빼면 overflow 위험이 줄어듭니다.
- Softmax는 한 행의 모든 logit에 같은 상수를 더하거나 빼도 결과가 같습니다.

Backward:

이 과제에서는 Softmax와 Cross Entropy를 합친 gradient를 `train()`에서 직접 만듭니다. 따라서 `Softmax.backward(dout)`는 받은 `dout`을 그대로 반환하는 구조입니다.

## Cross Entropy Loss

입력:

```text
y_pred: (N, C), softmax 확률
y_true: (N,), 정수 label
```

Forward:

```text
correct_probs = y_pred[np.arange(N), y_true]
loss = -mean(log(correct_probs))
```

수치 안정성:

```text
correct_probs = clip(correct_probs, eps, 1.0)
```

권장 `eps`:

```text
1e-7 또는 1e-12
```

Softmax + Cross Entropy combined gradient:

```text
dout = y_pred.copy()
dout[np.arange(N), y_true] -= 1
dout /= N
```

Shape:

```text
dout: (N, C)
```

## SGD

Update:

```text
params[key] -= lr * grads[key]
```

주의:

- `params`를 in-place로 바꿔야 모델의 layer가 참조하는 배열도 같이 업데이트됩니다.
- 모든 key에 대해 `params[key].shape == grads[key].shape`여야 합니다.

## Adam

일반적인 Adam update:

```text
t = t + 1
m = beta1 * m + (1 - beta1) * grad
v = beta2 * v + (1 - beta2) * grad^2
m_hat = m / (1 - beta1^t)
v_hat = v / (1 - beta2^t)
param -= lr * m_hat / (sqrt(v_hat) + eps)
```

일반적으로 쓰는 기본값:

```text
beta1 = 0.9
beta2 = 0.999
eps = 1e-8
```

현재 `Adam.__init__`에는 `lr`, `m`, `v`, `t`만 있습니다. 구현할 때 `beta1`, `beta2`, `eps`를 `update` 안의 지역 상수로 둘 수 있습니다.

## BatchNorm

Forward train mode:

```text
mu = mean(x, axis=0)
var = mean((x - mu)^2, axis=0)
std = sqrt(var + eps)
x_hat = (x - mu) / std
out = gamma * x_hat + beta
```

Running statistics:

```text
running_mean = momentum * running_mean + (1 - momentum) * mu
running_var = momentum * running_var + (1 - momentum) * var
```

Forward test mode:

```text
x_hat = (x - running_mean) / sqrt(running_var + eps)
out = gamma * x_hat + beta
```

Shape:

```text
x:            (N, H)
gamma, beta:  (H,)
out:          (N, H)
```

Backward values:

```text
dbeta = sum(dout, axis=0)
dgamma = sum(dout * x_hat, axis=0)
```

Compact backward formula:

```text
N = dout.shape[0]
std = sqrt(var + eps)
dx = (gamma / (N * std)) * (N * dout - dbeta - x_hat * dgamma)
```

주의:

- 이 compact formula는 forward train mode에서 저장한 `x_hat`, `var`, `gamma`가 필요합니다.
- `dgamma.shape == gamma.shape`
- `dbeta.shape == beta.shape`
- `dx.shape == x.shape`

## Dropout

현재 템플릿이 요구하는 기본 dropout 방식:

Train mode:

```text
mask = random values > drop_ratio
out = x * mask
```

Test mode:

```text
out = x * (1 - drop_ratio)
```

Backward:

```text
dx = dout * mask
```

주의:

- 이 과제의 test mode는 `x * (1 - drop_ratio)`를 기대합니다.
- inverted dropout 방식은 train 때 `x * mask / keep_prob`, test 때 `x`를 쓰지만, 현재 테스트와 TODO는 그 방식을 요구하지 않습니다.

## NeuralNetwork 권장 구조

과제 설명과 `src/network.py` TODO 기준 권장 구조:

```text
784 -> 512 -> 256 -> 10
```

은닉층 기본 순서:

```text
Affine -> BatchNorm -> ReLU -> Dropout
```

마지막 출력층:

```text
Affine -> Softmax
```

예상 params key 예시:

```text
W1, b1
gamma1, beta1
W2, b2
gamma2, beta2
W3, b3
```

단, `use_batchnorm=False`이면 `gamma`, `beta`는 만들지 않을 수 있습니다. 이 경우에도 `grads`는 `params`와 같은 key를 가져야 합니다.

## Weight Initialization

He initialization, ReLU와 함께 자주 사용:

```text
W = random_normal * sqrt(2 / fan_in)
```

Xavier initialization:

```text
W = random_normal * sqrt(1 / fan_in)
```

또는:

```text
W = random_normal * sqrt(2 / (fan_in + fan_out))
```

과제는 He 또는 Xavier 중 선택을 허용합니다. ReLU 기반 MLP에서는 He 초기화가 자연스럽습니다.

