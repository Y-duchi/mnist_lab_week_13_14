# Source Of Truth

이 문서는 현재 프로젝트에서 직접 확인한 사실만 정리합니다. ChatGPT에 질문할 때 이 문서를 함께 주면, 답변이 과제 요구와 어긋나는 일을 줄일 수 있습니다.

## 저장소 기준

- 저장소 경로: `/Users/yeoduchi/Documents/mnist_lab_week_13_14`
- 현재 브랜치: `main`
- GitHub default branch: `main`
- 원격 저장소: `https://github.com/Y-duchi/mnist_lab_week_13_14.git`

## 과제 목표

- PyTorch, TensorFlow 같은 딥러닝 프레임워크 없이 NumPy 중심으로 구현합니다.
- 최종 목표는 MNIST 필기체 숫자 분류기입니다.
- 테스트 정확도 목표는 권장 97% 이상, 최소 95% 이상입니다.
- 정확도만 맞히는 것이 아니라 Forward, Loss, Backward, Optimizer update 흐름을 설명할 수 있어야 합니다.

## 허용 및 금지 라이브러리

허용:

- `numpy`
- `math`
- `random`
- `time`
- `matplotlib`

과제 설명상 금지:

- PyTorch
- TensorFlow
- scikit-learn
- pandas
- 그 외 허용 목록에 없는 라이브러리

주의: 저장소의 `requirements.txt`에는 테스트 실행을 위한 `pytest`가 있을 수 있습니다. 구현 코드의 딥러닝 로직에는 허용 라이브러리만 사용합니다.

## 데이터 기준

파일: `src/data.py`

- `load_mnist(data_dir="data")`는 구현되어 있습니다.
- `data/mnist.npz`가 있으면 로컬 파일을 사용합니다.
- 없으면 `https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz`에서 다운로드합니다.
- 반환 형태:
  - `x_train`: `(60000, 784)`, `float32`, 0~1 정규화
  - `y_train`: `(60000,)`
  - `x_test`: `(10000, 784)`, `float32`, 0~1 정규화
  - `y_test`: `(10000,)`

## 구현 대상 파일

- `src/activations.py`
  - `ReLU.forward`
  - `ReLU.backward`
  - `Softmax.forward`
  - `Softmax.backward`

- `src/layers.py`
  - `Affine.forward`
  - `Affine.backward`
  - `BatchNorm.forward`
  - `BatchNorm.backward`
  - `Dropout.forward`
  - `Dropout.backward`

- `src/losses.py`
  - `cross_entropy_loss`

- `src/optimizers.py`
  - `SGD.update`
  - `Adam.update`

- `src/network.py`
  - `NeuralNetwork.__init__`
  - `NeuralNetwork.forward`
  - `NeuralNetwork.backward`

- `src/training.py`
  - `train`

## 테스트가 확인하는 계약

테스트는 최소 동작을 확인합니다. 테스트 통과가 곧 95% 이상 정확도를 보장하지는 않습니다.

### Step 1: ReLU

파일: `tests/test_relu.py`

- 양수 입력은 그대로 출력해야 합니다.
- 음수와 0은 0이 되어야 합니다.
- backward에서는 forward 때 `x <= 0`이었던 위치의 gradient가 0이어야 합니다.

### Step 2: Softmax

파일: `tests/test_softmax.py`

- 출력의 각 행 합은 1이어야 합니다.
- 확률은 0 이상 1 이하 범위여야 합니다.
- backward는 입력 gradient와 같은 shape를 반환해야 합니다.

### Step 3: Affine

파일: `tests/test_affine.py`

- forward는 `x @ W + b`를 계산해야 합니다.
- 출력 shape는 `(batch_size, output_dim)`이어야 합니다.
- backward는 `dx`, `dW`, `db`를 올바른 shape로 만들어야 합니다.

### Step 4: Cross Entropy

파일: `tests/test_cross_entropy_loss.py`

- batch 평균 loss 하나를 scalar로 반환해야 합니다.
- 정답 클래스 확률이 1에 가까우면 loss는 0에 가까워야 합니다.

### Step 5: SGD

파일: `tests/test_sgd.py`

- `params[key]`를 gradient 반대 방향으로 `lr`만큼 in-place update해야 합니다.

### Step 6: Adam

파일: `tests/test_adam.py`

- update 후 파라미터 값이 실제로 바뀌어야 합니다.
- 소스 TODO는 이동평균 `m`, `v`와 bias correction 사용을 요구합니다.

### Step 7: NeuralNetwork

파일: `tests/test_neural_network.py`

- `forward(x, train=True)`는 `(batch_size, 10)` 확률을 반환해야 합니다.
- `params` dict가 있어야 합니다.
- `backward(dout)` 후 `grads` dict가 있어야 하며, `len(grads) == len(params)`여야 합니다.

### Step 8: BatchNorm

파일: `tests/test_batchnorm.py`

- forward 출력 shape가 입력과 같아야 합니다.
- backward의 `dx` shape가 입력과 같아야 합니다.
- 소스 TODO는 train/test 통계 분리와 running 통계 갱신을 요구합니다.

### Step 9: Dropout

파일: `tests/test_dropout.py`

- train mode forward는 입력 shape를 유지해야 합니다.
- test mode forward는 `x * (1 - drop_ratio)`를 반환해야 합니다.
- 이 템플릿은 "학습 때 mask 적용, 추론 때 keep ratio로 scale"하는 기본 dropout 방식을 기대합니다.

### Step 10: train

파일: `tests/test_training.py`

- `train(...)`은 epoch별 loss history list를 반환해야 합니다.
- `epochs=1`이면 길이 1의 list여야 합니다.
- loss 값은 0 이상이어야 합니다.

### evaluate

파일: `tests/test_evaluate.py`

- `evaluate(model, x, y)`는 정확도와 파라미터 수를 반환합니다.
- 정확도는 0~100 범위여야 합니다.
- 파라미터 수는 양수여야 합니다.

## 현재 코드에서 이미 구현된 함수

- `src/data.py`의 `load_mnist`
- `src/training.py`의 `evaluate`
- `src/training.py`의 `plot_loss_history`

## 중요한 구분

테스트에서 확인된 사실:

- shape, 기본 계산, 최소 반환값 위주로 확인합니다.

정확도 달성을 위한 권장:

- 안정적인 Softmax와 Cross Entropy
- He 또는 Xavier 초기화
- BatchNorm running mean/variance 갱신
- Dropout의 train/test 분리
- Adam optimizer
- loss가 epoch마다 줄어드는지 기록

