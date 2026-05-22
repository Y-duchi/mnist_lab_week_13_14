# ChatGPT Context Pack

아래 내용을 ChatGPT에 붙여 넣고 질문하면, 답변이 현재 과제와 코드 구조에서 벗어나는 일을 줄일 수 있습니다.

```text
나는 NumPy만으로 MNIST 필기체 숫자 분류기를 구현하는 과제를 하고 있다.

절대 지켜야 할 규칙:
- PyTorch, TensorFlow, scikit-learn, pandas를 쓰지 않는다.
- 구현 코드에는 numpy, math, random, time, matplotlib만 사용한다.
- pytest는 테스트 실행용으로만 사용한다.
- 모르는 사실은 추측하지 말고 "현재 첨부 자료만으로는 확인할 수 없다"고 말한다.
- 과제 요구사항, 현재 src TODO, tests에서 확인되는 사실, 일반적인 신경망 지식을 구분해서 답한다.
- 책 내용을 그대로 인용하거나 복제하지 말고, 개념을 내 말로 설명한다.
- 내가 학습 중이므로 바로 최종 답안 전체를 던지기보다, 수식, shape, 흐름을 먼저 설명한다.
- 내가 코드 작성을 요청하면 현재 파일 구조와 테스트 기준에 맞춰 최소한의 코드만 제안한다.
- 허용되지 않은 라이브러리나 API를 권하지 않는다.

현재 프로젝트 구조:
- src/data.py: load_mnist 구현됨
- src/activations.py: ReLU, Softmax TODO
- src/layers.py: Affine, BatchNorm, Dropout TODO
- src/losses.py: cross_entropy_loss TODO
- src/optimizers.py: SGD, Adam TODO
- src/network.py: NeuralNetwork TODO
- src/training.py: train TODO, evaluate와 plot_loss_history는 구현됨
- tests/: 각 단계별 pytest가 있음

데이터:
- MNIST 이미지 입력은 28x28을 펼친 784차원 벡터다.
- x_train shape는 (60000, 784), x_test shape는 (10000, 784)다.
- 입력값은 0~1 범위로 정규화된다.
- label은 0~9 정수다.

미니배치 학습 순서:
1. y_pred = model.forward(x_batch, train=True)
2. loss = cross_entropy_loss(y_pred, y_batch)
3. dout = y_pred.copy()
4. dout[np.arange(batch_size), y_batch] -= 1
5. dout /= batch_size
6. model.backward(dout)
7. optimizer.update(model.params, model.grads)

구현 순서와 테스트:
1. ReLU.forward/backward: pytest tests/test_relu.py -v
2. Softmax.forward/backward: pytest tests/test_softmax.py -v
3. Affine.forward/backward: pytest tests/test_affine.py -v
4. cross_entropy_loss: pytest tests/test_cross_entropy_loss.py -v
5. SGD.update: pytest tests/test_sgd.py -v
6. Adam.update: pytest tests/test_adam.py -v
7. NeuralNetwork: pytest tests/test_neural_network.py -v
8. BatchNorm: pytest tests/test_batchnorm.py -v
9. Dropout: pytest tests/test_dropout.py -v
10. train: pytest tests/test_training.py -v
전체 테스트: pytest tests/ -v

테스트에서 확인되는 핵심 계약:
- ReLU: x <= 0 위치는 forward 출력 0, backward gradient 0
- Softmax: 각 행 확률 합은 1, backward는 dout shape 유지
- Affine: forward는 x @ W + b, backward는 dx/dW/db shape 유지
- Cross Entropy: batch 평균 scalar loss, 정답 확률이 1에 가까우면 loss는 0에 가까움
- SGD: params[key] -= lr * grads[key]
- Adam: m, v 이동평균과 bias correction을 사용해 params를 바꿈
- NeuralNetwork: forward 출력 shape는 (batch, 10), params와 grads dict 필요
- BatchNorm: forward/backward shape 유지, train/test 통계 분리 필요
- Dropout: train에서는 mask 적용, test에서는 x * (1 - drop_ratio)
- train: epoch별 loss_history list 반환

권장 모델 구조:
입력 784 -> Affine(512) -> BatchNorm -> ReLU -> Dropout -> Affine(256) -> BatchNorm -> ReLU -> Dropout -> Affine(10) -> Softmax

목표:
- 먼저 각 테스트를 통과시킨다.
- 그 다음 loss가 줄어드는지 확인한다.
- 최종적으로 test accuracy 최소 95%, 권장 97% 이상을 목표로 실험한다.
```

## 질문 템플릿

### 개념 설명 요청

```text
위 과제 기준에서 [개념 이름]을 설명해줘.
반드시 다음 순서로 답해줘:
1. 이 개념이 과제의 어느 파일/함수와 연결되는지
2. forward에서 하는 일
3. backward에서 필요한 gradient
4. shape 예시
5. 내가 자주 실수할 부분
추측이 필요한 부분은 추측이라고 표시해줘.
```

### 코드 리뷰 요청

```text
위 과제 기준에서 내 구현을 리뷰해줘.
관점:
- 테스트 계약 위반
- shape 오류
- train/test mode 오류
- 수치 안정성 문제
- 허용 라이브러리 위반
- 정확도에 영향을 줄 수 있는 문제
확실한 문제와 개선 제안을 구분해줘.
```

### 디버깅 요청

```text
아래 에러를 현재 과제 구조 기준으로 디버깅해줘.
답변 순서:
1. 에러가 어느 단계에서 난 것 같은지
2. 가장 가능성 높은 원인 3개
3. 확인할 print/debug 코드
4. 수정 방향
5. 여전히 모르는 부분
```

### 구현 힌트 요청

```text
정답 코드를 한 번에 주지 말고 힌트부터 줘.
내가 구현하려는 함수: [함수명]
현재 헷갈리는 점: [내용]
필요한 것은 수식, shape, 작은 NumPy 예시야.
```

## 답변 검증 질문

ChatGPT 답변을 받은 뒤 아래 질문으로 검증합니다.

- 이 답변이 현재 `src/` 파일의 함수 이름과 맞는가?
- 이 답변이 `tests/`에서 확인하는 동작과 맞는가?
- 허용 라이브러리 밖의 도구를 권하지 않았는가?
- BatchNorm과 Dropout의 train/test mode를 구분했는가?
- `params`와 `grads`의 key/shape가 같은 구조인가?
- "테스트가 요구하는 것"과 "정확도를 위한 권장"을 구분했는가?
- 불확실한 내용을 확정처럼 말하지 않았는가?

