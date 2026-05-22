# Learning Roadmap

이 로드맵은 책을 읽으면서 현재 과제 구현 순서와 연결하기 위한 자료입니다. 책의 문장을 옮기는 대신, 과제 구현에 필요한 개념을 독립적으로 정리합니다.

## 전체 전략

처음부터 97% 정확도를 목표로 잡지 않습니다. 아래 순서가 더 안전합니다.

1. 작은 단위 테스트 통과
2. `Forward -> Loss -> Backward -> Update` 흐름 이해
3. 작은 랜덤 데이터로 학습 루프 동작 확인
4. MNIST 일부 데이터로 loss 감소 확인
5. 전체 학습 후 정확도 향상 실험
6. REPORT에 실험 설정과 결과 기록

## 책을 읽을 때 연결할 질문

### NumPy와 배열 계산

구현 연결:

- `x @ W + b`
- `np.sum(..., axis=0)`
- `np.max(..., axis=1, keepdims=True)`
- broadcasting
- boolean mask

읽으면서 답해야 할 질문:

- `(N, D) @ (D, H)`의 결과 shape는 무엇인가?
- `keepdims=True`를 쓰면 broadcasting에서 어떤 장점이 있는가?
- 원본 배열을 직접 바꾸는 in-place update가 optimizer에서 왜 필요한가?

### 퍼셉트론과 선형 변환

구현 연결:

- `Affine.forward`
- `Affine.backward`
- `W`, `b`, `dW`, `db`, `dx`

읽으면서 답해야 할 질문:

- 선형 변환만 쌓으면 왜 복잡한 패턴을 잘 표현하지 못하는가?
- MNIST 입력 784개 숫자가 은닉층 차원으로 바뀌는 과정은 어떤 행렬 곱인가?

### 활성화 함수

구현 연결:

- `ReLU.forward`
- `ReLU.backward`

읽으면서 답해야 할 질문:

- ReLU는 왜 은닉층에 비선형성을 추가하는가?
- forward 때 저장한 mask가 backward에서 왜 필요한가?
- `x == 0`일 때 이 과제에서는 gradient를 어떻게 처리해야 하는가?

### 출력층과 확률

구현 연결:

- `Softmax.forward`
- `cross_entropy_loss`
- `train` 안의 combined gradient

읽으면서 답해야 할 질문:

- Softmax 출력의 각 행 합이 왜 1이어야 하는가?
- logit에 같은 값을 모두 더해도 softmax 결과가 같다는 성질이 수치 안정성과 어떻게 연결되는가?
- 정답 클래스 확률이 작을수록 cross entropy loss가 커지는 이유는 무엇인가?

### 학습과 gradient

구현 연결:

- `model.backward(dout)`
- `SGD.update`
- `Adam.update`
- `train`

읽으면서 답해야 할 질문:

- loss가 줄어드는 방향과 gradient의 방향은 어떤 관계인가?
- `params`와 `grads`의 key와 shape가 같아야 하는 이유는 무엇인가?
- 한 epoch와 한 mini-batch는 어떻게 다른가?

### Backpropagation

구현 연결:

- `Affine.backward`
- `ReLU.backward`
- `BatchNorm.backward`
- `NeuralNetwork.backward`

읽으면서 답해야 할 질문:

- 각 layer는 backward에서 무엇을 입력받고 무엇을 반환하는가?
- 왜 forward 때의 중간값을 저장해야 backward를 계산할 수 있는가?
- layer를 역순으로 통과해야 하는 이유는 무엇인가?

### 학습 관련 기법

구현 연결:

- He 또는 Xavier initialization
- BatchNorm
- Dropout
- Adam

읽으면서 답해야 할 질문:

- ReLU를 쓸 때 He 초기화가 왜 자주 쓰이는가?
- BatchNorm은 train mode와 test mode에서 왜 다른 통계를 쓰는가?
- Dropout은 train mode와 test mode에서 왜 다르게 동작하는가?
- Adam의 `m`, `v`, bias correction은 무엇을 보정하는가?

## 권장 진행 일정

### 1일차: 배열과 기본 layer

- `ReLU`
- `Softmax`
- `Affine`
- `cross_entropy_loss`

확인:

```bash
pytest tests/test_relu.py -v
pytest tests/test_softmax.py -v
pytest tests/test_affine.py -v
pytest tests/test_cross_entropy_loss.py -v
```

### 2일차: optimizer와 네트워크 조립

- `SGD`
- `Adam`
- `NeuralNetwork.__init__`
- `NeuralNetwork.forward`
- `NeuralNetwork.backward`

확인:

```bash
pytest tests/test_sgd.py -v
pytest tests/test_adam.py -v
pytest tests/test_neural_network.py -v
```

### 3일차: 정규화, 규제, 학습 루프

- `BatchNorm`
- `Dropout`
- `train`
- `evaluate`

확인:

```bash
pytest tests/test_batchnorm.py -v
pytest tests/test_dropout.py -v
pytest tests/test_training.py -v
pytest tests/test_evaluate.py -v
```

### 4일차 이후: 정확도 실험

- 전체 테스트 통과
- 작은 epoch로 loss 감소 확인
- Adam lr `0.001`, `0.0005`, `0.0001` 비교
- Dropout ratio `0.2`, `0.3`, `0.5` 비교
- BatchNorm 사용 여부 비교
- REPORT 작성

확인:

```bash
pytest tests/ -v
```

## 매 단계 설명 연습

각 구현을 끝낼 때 팀원에게 아래 문장을 완성해서 설명합니다.

- 이 layer의 forward 입력 shape는 `...`이고 출력 shape는 `...`이다.
- backward는 다음 layer에서 `...`를 받고 이전 layer로 `...`를 넘긴다.
- 이 layer가 저장해야 하는 중간값은 `...`이다.
- 이 layer의 학습 파라미터는 `...`이고 gradient는 `...`이다.
- 이 구현에서 수치 안정성을 위해 처리한 부분은 `...`이다.

