# MNIST NumPy 신경망 학습 가이드

이 폴더는 현재 과제 저장소를 기준으로 만든 학습 자료입니다. 폴더명 `stydy_guide`는 사용자가 요청한 이름을 그대로 따랐습니다.

목표는 두 가지입니다.

1. 책을 읽으면서 구현 순서, 수식, shape를 놓치지 않도록 돕기
2. ChatGPT에게 첨부할 때 과제 요구사항과 현재 코드 기준을 명확히 제공해 환각을 줄이기

## 읽는 순서

1. `00_source_of_truth.md`
   - 현재 저장소에서 확인한 사실만 모아 둔 기준 문서입니다.
   - README, `src/`, `tests/`를 기준으로 작성했습니다.

2. `01_learning_roadmap.md`
   - 책을 읽으며 어떤 개념을 어느 구현 단계와 연결할지 정리했습니다.
   - 정확도보다 먼저 테스트와 loss 감소를 확인하는 흐름입니다.

3. `02_math_numpy_reference.md`
   - ReLU, Softmax, Cross Entropy, Affine, BatchNorm, Dropout, SGD, Adam의 수식과 shape 기준입니다.
   - 구현할 때 가장 자주 확인할 문서입니다.

4. `03_implementation_checklist.md`
   - TODO 파일별 구현 순서와 테스트 명령입니다.
   - 각 단계에서 확인해야 할 조건을 체크리스트로 적었습니다.

5. `04_chatgpt_context_pack.md`
   - ChatGPT에 그대로 붙여 넣기 위한 컨텍스트 팩입니다.
   - "모르면 모른다고 말하기", "허용 라이브러리 외 사용 금지", "출처 구분" 규칙이 포함되어 있습니다.

6. `05_debugging_and_experiment_log.md`
   - 막혔을 때 확인할 증상별 점검표와 실험 기록 양식입니다.

## 환각 방지 원칙

- 과제 요구사항은 `README.md`, 사용자가 제공한 과제 설명, 현재 `src/`와 `tests/`가 1차 기준입니다.
- 책 내용은 이해를 돕는 배경으로만 사용하고, 책 문장을 그대로 복제하지 않습니다.
- 구현 세부사항을 말할 때는 "테스트에서 확인된 사실"과 "정확도 달성을 위한 권장"을 구분합니다.
- PyTorch, TensorFlow, scikit-learn, pandas 등은 사용하지 않습니다.
- NumPy 구현에서 shape가 불명확하면 먼저 작은 배열로 검산합니다.

## 현재 과제의 핵심 흐름

한 미니배치마다 반드시 이 순서로 생각합니다.

```text
Forward -> Loss -> Backward -> Optimizer update
```

출력층에서는 Softmax와 Cross Entropy를 합친 gradient를 직접 만듭니다.

```text
dout = y_pred.copy()
dout[range(batch_size), y_batch] -= 1
dout /= batch_size
```

그 다음 `model.backward(dout)`과 `optimizer.update(model.params, model.grads)`로 이어집니다.

