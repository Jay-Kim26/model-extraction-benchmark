# Questions for New Attacks (SwiftThief, Black-box Dissector, CloudLeak)

## SwiftThief
1. 초기 쿼리 세트 Q 구성 방식: 랜덤/entropy/고정 비율 중 무엇?
2. 매 round 쿼리 수 k 또는 스텝 규칙: 고정 k? 아니면 budget 비율?
3. Q/U 데이터 분할은 seed pool 기반으로 구성하면 될까?
4. `L_total = L_contrastive + λ3 * L_matching`에서 `L_matching`을 CE loss로 보면 되는지?
5. 희소 클래스 우선으로 전환되는 기준(몇 라운드 이후? entropy가 떨어진 시점?)
6. FGSM regularizer 적용 시점: 학습시에만 적용? 쿼리는 원본?

## Black-box Dissector
1. CAM-driven erasing 쿼리 방식: 원본 쿼리 후 erasing 재쿼리로 2배 예산 소진 맞나?
2. Erasing 영역 크기/threshold 기준: 비율 고정? Grad-CAM 상위 %?
3. "확률 변화가 가장 큰 이미지 선택"의 단위: 풀에서 top-k 뽑아서 쿼리? 아니면 per-sample 방식?
4. Self-KD random erasing의 N(반복 횟수) 값?
5. Grad-CAM은 substitute 기준? victim 기준? (victim 내부 접근 불가 시 substitute 기준으로 가야 함)

## CloudLeak
1. FeatureFool 최적화 파라미터: L-BFGS iterations, λ, margin M(0.5 고정?)
2. Victim feature 접근 불가하면 substitute feature로 대체하는 게 맞나?
3. 매 라운드 synthetic 샘플 생성 개수 vs 실제 쿼리 개수(k)
4. Pretrained backbone: torchvision 모델 freeze + FC만 학습으로 가도 되는지?
5. FeatureFool 입력 x는 랜덤 풀 샘플? 아니면 active 선택?

## 공통
1. 각 공격의 기본 config에 넣을 하이퍼파라미터 값(명시된 항목) 확정 필요
2. 테스트 스코프:
   - SwiftThief: contrastive loss 계산 + rare-class sampling 전환
   - Black-box Dissector: erasing + pseudo-label 평균 로직
   - CloudLeak: FeatureFool 샘플 생성/선택 로직
3. 실험용 config 파일 3개 추가해도 되는지? (`configs/swiftthief_experiment.yaml` 등)

---

## 추가 질문 (답변 바탕)

### SwiftThief
1. 희소 클래스 우선 전환 조건 수식 해석:
   - `B - |Q| ≤ N_R · (μ - μ_R)`에서 N_R, μ, μ_R의 구체적 계산 방법은?
   - μ = 전체 클래스 평균 샘플 수, μ_R = 희소 클래스 평균 샘플 수로 이해하면 되는지?
2. 총 반복 횟수 I 결정 기준: budget/1000 기준 checkpoints? 아니면 고정 값?

### Black-box Dissector
1. Erasing 후 pool에서 예산만큼 선택하는 기준: 정렬 후 top-k 방식인지?
2. Erasing 변형 생성 수: 이미지당 몇 개의 variant를 생성하는지?

### 공통
1. 실험용 config 파일 3개 추가: `configs/swiftthief_experiment.yaml`, `configs/dissector_experiment.yaml`, `configs/cloudleak_experiment.yaml`



제공된 문헌(SwiftThief, Black-box Dissector, CloudLeak)을 바탕으로 구현에 필요한 구체적인 정보를 답변드립니다.

### ## SwiftThief

1.  **초기 쿼리 세트 Q 구성 방식:** **균등 무작위 샘플링 (Uniform Random Sampling)**을 사용합니다. 알고리즘 1의 4-5번째 줄에 따르면 첫 번째 반복($k=0$)에서는 무작위 선택 전략을 사용한다고 명시되어 있습니다.
2.  **매 round 쿼리 수 k 또는 스텝 규칙:** 전체 예산 $B$와 총 반복 횟수 $I$에 따라 매 라운드 **고정된 수의 샘플 $\lceil B/I \rceil$**를 선택합니다.
3.  **Q/U 데이터 분할:** 네, 가능합니다. 대리 데이터셋(Surrogate dataset) $S$를 준비한 뒤, 초기에는 쿼리된 집합 $Q = \emptyset$, 쿼리되지 않은 집합 $U = S$로 설정하고 진행함에 따라 샘플을 $U$에서 $Q$로 이동시킵니다,.
4.  **`L_matching`의 정체:** 네, **Cross-Entropy (CE) Loss**입니다. 수식 (6)에서 $L_m$은 피해 모델의 응답 $y_i$와 클론 모델의 출력 $f_A(x_i)$ 간의 교차 엔트로피로 정의됩니다.
5.  **희소 클래스 우선 전환 기준:** 남은 예산이 희소 클래스의 불균형을 해소하기 위한 임계값보다 작아질 때 전환됩니다. 구체적인 조건은 $B - |Q| \le N_R \cdot (\mu - \mu_R)$이며, 여기서 $N_R$은 희소 클래스 개수, $\mu$는 전체 클래스 평균 샘플 수, $\mu_R$은 희소 클래스 평균 샘플 수입니다.
6.  **FGSM Regularizer 적용 시점:** **학습(가중치 업데이트) 시에만 적용**됩니다. $L_{reg}^c$는 대조 학습 손실 함수의 일부로 사용되어 클론 모델의 표현 학습을 돕습니다. 피해 모델에 보내는 쿼리(Line 12)는 $U$에서 선택된 **원본 샘플** $x$입니다.

### ## Black-box Dissector

1.  **CAM-driven erasing 쿼리 방식:** 네, **2배의 예산 소진 개념이 맞습니다.** 알고리즘 2에 따르면, 먼저 $D_U$에서 샘플을 선택하여 쿼리(Step 1)한 후, 해당 데이터($D_T$)에서 Erasing을 수행하고 다시 피해 모델에 쿼리하여 라벨을 얻는 과정(Step 2.1)을 거칩니다. 두 단계 모두 예산($q$)을 증가시킵니다.
2.  **Erasing 영역 크기/threshold:** 고정된 비율이 아니라 **최적화/선택 방식**입니다. 식 (4)와 (5)에 따라, 지워진 이미지들 중 대체 모델의 신뢰도(Maximum Softmax Probability)가 가장 높거나 원래 라벨과의 차이가 가장 큰 변형을 선택합니다,.
3.  **"확률 변화가 가장 큰 이미지 선택"의 단위:** **Per-sample 방식**으로 각 이미지마다 가장 효과적인 Erasing 변형을 찾습니다. 그 후 전체 풀에서 예산(budget)만큼 샘플을 선택합니다,.
4.  **Self-KD random erasing의 N 값:** 제공된 문헌 내에 구체적인 정수값(예: 10, 20 등)은 **명시되어 있지 않습니다**. 식 (6)에서 $N$번 반복하여 평균을 낸다고만 기술되어 있습니다.
5.  **Grad-CAM 기준:** **Substitute (대체) 모델 기준**입니다. 피해 모델은 블랙박스이므로 내부 그래디언트에 접근할 수 없어, 대체 모델을 사용하여 근사적인 Attention map을 계산합니다.

### ## CloudLeak

1.  **FeatureFool 최적화 파라미터:**
    *   **Margin $M$:** 문헌에서 **$\alpha = 0.5$**로 설정한다고 명시되어 있습니다 (식 11 관련).
    *   **$\lambda$:** 실험적으로 **가장 작은 값(smallest value)**을 선택하는 것이 최적이라고 언급되어 있습니다.
    *   **L-BFGS iterations:** 문헌 내에 구체적인 반복 횟수는 명시되지 않았습니다.
2.  **Victim feature 접근 불가 시:** 네, **Substitute 모델의 feature로 대체**합니다. 공격자는 초기 데이터셋 $X_0$에 대해 **Substitute 모델을 대상**으로 적대적 예제를 생성(FeatureFool 등 사용)하고, 이를 Victim에 쿼리하여 라벨을 얻는 방식을 사용합니다.
3.  **Synthetic 샘플 생성 vs 실제 쿼리 개수:** 문헌상에서는 생성된 악성 예제(Malicious Examples)를 사용하여 피해 모델을 쿼리하고 라벨을 얻어 합성 데이터셋을 구축한다고 설명하므로, **생성된 유효 샘플 수와 실제 쿼리 수는 1:1 관계**로 볼 수 있습니다.
4.  **Pretrained backbone 학습:** 네, 가능합니다. 문헌에서는 사전 학습된 모델(예: VGG19)을 전이 학습 아키텍처로 사용하며, **이전의 합성곱 층(convolution layers)은 고정(frozen)하고 마지막 몇 개의 완전 연결 층(fully connected layers)만 재학습**한다고 명시되어 있습니다.
5.  **FeatureFool 입력 x:** **랜덤 풀 샘플**입니다. 알고리즘의 첫 단계에서 타겟 도메인으로부터 작은 세트 $X_0$를 **무작위로 샘플링(randomly sample)**하여 초기 데이터셋으로 사용하고, 이 데이터에 대해 적대적 예제를 생성합니다.


제공된 문헌(SwiftThief, Black-box Dissector)을 바탕으로 질문하신 구현 세부 사항에 대해 답변드립니다.

### ### SwiftThief

**1. 희소 클래스 우선 전환 조건 수식 해석**

질문하신 수식 `B - |Q| ≤ N_R · (μ - μ_R)`의 변수 정의와 계산 방법은 논문의 4.2절 "Prioritization of Rarely Queried Classes in Sampling"에 명시된 내용과 정확히 일치합니다. 이해하신 내용이 맞으며, 구체적인 정의는 다음과 같습니다.

*   **희소 클래스(Rarely Represented Class)의 정의:** 현재 쿼리된 집합 $Q$ 내에서, 해당 클래스의 샘플 수가 **모든 클래스의 평균 샘플 수($\mu$)보다 적은 클래스**를 의미합니다.
*   **변수 계산 방법:**
    *   **$|Q|$:** 현재까지 쿼리된 샘플의 총개수입니다.
    *   **$N_R$:** 위에서 정의한 **희소 클래스의 개수(count)**입니다.
    *   **$\mu$ (전체 평균):** 현재 $Q$에 있는 **모든 클래스**의 평균 샘플 수입니다. (즉, $|Q| / \text{전체 클래스 수 K}$).
    *   **$\mu_R$ (희소 클래스 평균):** **희소 클래스로 식별된 클래스들**의 샘플 수 평균입니다.
*   **수식의 의미:** 남은 예산($B - |Q|$)이 희소 클래스들의 부족한 샘플 수를 전체 평균 수준으로 채워주는 데 필요한 양($N_R \cdot (\mu - \mu_R)$)보다 작거나 같아지면, 즉시 희소 클래스 우선 샘플링으로 전환하여 불균형을 해소하겠다는 의미입니다.

**2. 총 반복 횟수 I 결정 기준**

총 반복 횟수 $I$는 예산에 따라 동적으로 변하는 값이 아니라 **고정된 하이퍼파라미터**입니다.

*   **설정 값:** 실험 설정(Section 5.1)에서 **$I = 10$**으로 고정하여 사용했습니다.
*   **배치 크기:** 전체 예산 $B$가 주어지면, 매 라운드마다 쿼리하는 샘플의 수(배치 크기)는 $\lceil B/I \rceil$로 계산됩니다. 실험에서는 $B=30,000$일 때 $I=10$을 사용하여 매 라운드 3,000개씩 쿼리하는 방식을 사용했습니다.

---

### ### Black-box Dissector

**1. Erasing 후 pool에서 예산만큼 선택하는 기준**

단순한 랜덤 선택이 아니라 **대체 모델(Substitute Model)의 신뢰도(Confidence)를 기준으로 정렬하여 상위 k개(Top-k)를 선택**하는 방식입니다.

*   **선택 과정:**
    1.  각 이미지 $x$에 대해 Grad-CAM 기반으로 Erasing을 수행하여 변형된 이미지(Variant)를 생성합니다.
    2.  이 변형된 이미지들을 대체 모델($\hat{f}$)에 입력합니다.
    3.  **MSP (Maximum Softmax Probability)** 값을 계산하여, 대체 모델이 가장 높은 확률로 예측한(가장 자신 있어 하는) 샘플들을 우선적으로 선택합니다.
*   **이유:** 대체 모델이 높은 확신을 가지는 변형 이미지가 피해 모델(Victim)에 쿼리되었을 때, 만약 라벨이 바뀐다면 매우 유용한 정보(Decision Boundary 근처 정보 등)를 얻을 수 있고, 바뀌지 않더라도 대체 모델의 Attention을 보정하는 데 유용하기 때문입니다.

**2. Erasing 변형 생성 수**

이미지당 **$N$개의 변형(Variant)을 생성**한 후, 그중 **가장 효과적인 1개**를 최종 후보로 남깁니다.

*   **생성 및 필터링 과정:**
    *   하나의 원본 이미지 $x$에 대해 CAM 기반 Erasing을 **$N$번 수행**합니다 ($i \in [N]$).
    *   생성된 $N$개의 변형 중, 원본 라벨(Hard-label)과 비교했을 때 **대체 모델의 예측 확률 분포 변화가 가장 큰(가장 헷갈려 하거나 라벨이 바뀔 가능성이 높은) 1개의 변형**을 선택합니다 (식 4: `arg max i \in [N]`).
    *   이 '최적의 변형' 1개가 해당 이미지를 대표하여 위에서 언급한 Pool 선택 과정(Top-k Selection)의 후보가 됩니다.
*   **$N$ 값:** 제공된 문헌의 발췌본에는 $N$의 구체적인 정수 값(예: 10, 20)이 명시되어 있지 않으나, 알고리즘적으로는 $N$번 반복하여 최적의 1개를 찾는 구조임이 명확합니다. (일반적으로 이러한 공격에서는 10~20 내외의 값을 사용합니다).