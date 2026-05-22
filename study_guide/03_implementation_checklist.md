# Implementation Checklist

이 체크리스트는 과제 구현 순서와 테스트 명령을 맞춘 것입니다. 한 단계씩 통과시키고 다음 단계로 넘어갑니다.

## 시작 전 확인

```bash
git status --short --branch
python --version
pip install -r requirements.txt
```

Python은 과제 기준상 3.11을 사용합니다.

## Step 1: ReLU

파일:

- `src/activations.py`

구현:

- `ReLU.forward`
- `ReLU.backward`

확인할 것:

- forward에서 `x > 0` mask를 저장했는가?
- `x <= 0` 위치가 0으로 바뀌는가?
- backward에서 mask가 false인 위치의 gradient가 0인가?

테스트:

```bash
pytest tests/test_relu.py -v
```

## Step 2: Softmax

파일:

- `src/activations.py`

구현:

- `Softmax.forward`
- `Softmax.backward`

확인할 것:

- `np.max(x, axis=1, keepdims=True)`로 row별 max를 뺐는가?
- 출력 각 행의 합이 1인가?
- 확률이 0~1 범위인가?
- backward는 이 과제 구조에서 `dout`을 그대로 반환하는가?

테스트:

```bash
pytest tests/test_softmax.py -v
```

## Step 3: Affine

파일:

- `src/layers.py`

구현:

- `Affine.forward`
- `Affine.backward`

확인할 것:

- forward에서 `self.x`를 저장했는가?
- `out = x @ W + b`인가?
- `dW = x.T @ dout`인가?
- `db = np.sum(dout, axis=0)`인가?
- `dx = dout @ W.T`인가?
- shape가 모두 맞는가?

테스트:

```bash
pytest tests/test_affine.py -v
```

## Step 4: Cross Entropy Loss

파일:

- `src/losses.py`

구현:

- `cross_entropy_loss`

확인할 것:

- 정답 label 위치의 확률만 골랐는가?
- `np.clip`으로 `log(0)`을 피했는가?
- batch 평균 scalar를 반환하는가?

테스트:

```bash
pytest tests/test_cross_entropy_loss.py -v
```

## Step 5: SGD

파일:

- `src/optimizers.py`

구현:

- `SGD.update`

확인할 것:

- 모든 key에 대해 update하는가?
- gradient 반대 방향으로 움직이는가?
- in-place update인가?

테스트:

```bash
pytest tests/test_sgd.py -v
```

## Step 6: Adam

파일:

- `src/optimizers.py`

구현:

- `Adam.update`

확인할 것:

- 첫 update 전에 `m[key]`, `v[key]`를 0 배열로 초기화하는가?
- `t`를 update마다 증가시키는가?
- `m`, `v` 이동평균을 계산하는가?
- bias correction을 적용하는가?
- `sqrt(v_hat) + eps`로 나누는가?

테스트:

```bash
pytest tests/test_adam.py -v
```

## Step 7: NeuralNetwork

파일:

- `src/network.py`

구현:

- `NeuralNetwork.__init__`
- `NeuralNetwork.forward`
- `NeuralNetwork.backward`

확인할 것:

- `params` dict가 있는가?
- `grads` dict가 `params`와 같은 key를 갖는가?
- layer 순서가 `OrderedDict`에 보존되는가?
- 은닉층은 `Affine -> BatchNorm -> ReLU -> Dropout` 순서인가?
- 출력층 뒤에 Softmax가 적용되는가?
- backward는 layer를 역순으로 통과하는가?
- Affine과 BatchNorm의 gradient를 `self.grads`에 모으는가?

테스트:

```bash
pytest tests/test_neural_network.py -v
```

## Step 8: BatchNorm

파일:

- `src/layers.py`

구현:

- `BatchNorm.forward`
- `BatchNorm.backward`

확인할 것:

- train mode에서 batch mean/var를 쓰는가?
- train mode에서 running mean/var를 갱신하는가?
- test mode에서 running mean/var를 쓰는가?
- backward에서 `dgamma`, `dbeta`, `dx` shape가 맞는가?

테스트:

```bash
pytest tests/test_batchnorm.py -v
```

## Step 9: Dropout

파일:

- `src/layers.py`

구현:

- `Dropout.forward`
- `Dropout.backward`

확인할 것:

- train mode에서 mask를 새로 만드는가?
- train mode output shape가 입력과 같은가?
- test mode에서 `x * (1 - drop_ratio)`를 반환하는가?
- backward에서 forward mask를 곱하는가?

테스트:

```bash
pytest tests/test_dropout.py -v
```

## Step 10: train

파일:

- `src/training.py`

구현:

- `train`

확인할 것:

- epoch마다 데이터를 섞는가?
- batch 단위로 자르는가?
- 각 batch에서 `model.forward(x_batch, train=True)`를 호출하는가?
- `cross_entropy_loss(y_pred, y_batch)`를 계산하는가?
- Softmax + Cross Entropy combined gradient를 만드는가?
- `model.backward(dout)`을 호출하는가?
- `optimizer.update(model.params, model.grads)`를 호출하는가?
- epoch별 평균 loss를 `loss_history`에 추가하는가?

테스트:

```bash
pytest tests/test_training.py -v
```

## 전체 테스트

```bash
pytest tests/ -v
```

## 정확도 실험 전 체크

- 전체 테스트가 통과했는가?
- 작은 데이터에서 loss가 줄어드는가?
- `model.predict`는 `train=False`로 동작하는가?
- BatchNorm running statistics가 학습 중 갱신되는가?
- Dropout은 평가 때 random mask를 만들지 않는가?
- `params`와 `grads`의 key, shape가 모두 맞는가?

## REPORT 기록 항목

- 모델 구조
- optimizer
- learning rate
- epochs
- batch size
- BatchNorm 사용 여부
- Dropout ratio
- weight initialization
- train loss
- test accuracy
- total parameter count
- 학습 시간
- 실패한 실험과 바꾼 점

