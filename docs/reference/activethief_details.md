1. ACTIVETHIEF 구현 정보 요약표
항목
상세 내용 및 출처
1. 알고리즘 절차
1. 초기화: Thief 데이터셋에서 무작위로 초기 시드 S 
0
​
  선택.<br>2. 쿼리: 피해 모델(Secret Model f)에 S 
i
​
 를 입력하여 레이블 D 
i
​
  획득.<br>3. 학습: 현재까지 수집된 모든 레이블 데이터 ∪D 
t
​
 로 대체 모델(Substitute Model  
f
~
​
 )을 처음부터(from scratch) 재학습.<br>4. 후보 평가: 남은 Thief 데이터(Pool)에 대해  
f
~
​
 로 예측 수행.<br>5. 선택: 능동 학습 전략(Entropy, K-center 등)을 사용해 다음 쿼리할 S 
i+1
​
  선택.<br>6. 반복: 예산 소진 시까지 2~5단계 반복,.
2. 쿼리 생성/선택 방식
Selection from Pool 방식 사용 (합성 데이터 생성 아님).<br>- Uncertainty: 엔트로피가 가장 높은 샘플 선택.<br>- K-center: 현재 선택된 센터들로부터 특징 공간(feature space) 상 거리가 가장 먼 샘플 선택 (Greedy).<br>- DFAL: DeepFool 섭동(perturbation)이 가장 작은(경계에 가까운) 샘플 선택.<br>- DFAL+K-center: DFAL로 ρ개 후보 선정 후, K-center로 k개 최종 선택.
3. 학습 루프 구조
Round-based Iterative Learning:<br>Query(S 
i
​
 ) → Labeling → Train Substitute( 
f
~
​
 ) → Predict Pool → Select(S 
i+1
​
 ) 순서로 진행,.<br>매 라운드마다 대체 모델을 처음부터 다시 학습함(re-training from scratch).
4. 하이퍼파라미터
- Optimizer: Adam.<br>- Batch Size: 150 (이미지), 50 (텍스트).<br>- Max Epochs: 1,000 (Early Stopping 적용).<br>- Patience: 100 epoch (이미지), 20 epoch (텍스트).<br>- Initial Seed: 전체 예산의 10%.<br>- Regularization: L2 (0.001), Dropout (0.1, CIFAR-10은 0.2).
5. 손실 함수 구성
문헌에 명시적인 수식(예: CrossEntropy)은 기재되지 않음. 단, 검증(Validation) 시 F1 Score를 사용하여 최적 모델을 선택한다고 명시.<br>선택 전략(DFAL)에서는 DeepFool 섭동 크기 α 
n
​
 =∣x 
n
​
 − 
x
^
  
n
​
 ∣ 
2
2
​
 를 최소화하는 기준을 사용.
6. 데이터/전처리
Thief Dataset: 레이블이 없는 공개 데이터(NNPD) 사용 (예: ImageNet downsampled, WikiText-2),.<br>Victim과의 관계: 피해 모델의 학습 데이터(Problem Domain)와 분포가 달라도 됨(Natural Non-Problem Domain).<br>전처리 세부 규칙은 명시되지 않았으나, 표준 입력 크기 조정 등을 가정함.
7. 모델 아키텍처
Student (Substitute):<br>- 이미지: Conv 블록(2×Conv, 1×Pool) ×l개 반복. (기본 l=3). 각 Conv 뒤 ReLU, Batchnorm 적용.<br>- 텍스트: Word2vec → CNN (Kim) 또는 RNN (GRU, 64 units).<br>Victim: 공격자는 Victim의 아키텍처를 정확히 몰라도 되며, 다른 아키텍처(CNN vs RNN)를 사용해도 추출 가능함.
8. 구현 주의사항
- 재학습 비용: 매 반복마다 모델을 초기화하고 재학습해야 하므로 연산 비용이 높음.<br>- Validation Set: 쿼리 예산의 20%를 검증용으로 할당하여 모델 선택에 사용.<br>- 탐지 회피: 합성 데이터가 아닌 자연 데이터(NNPD)를 쿼리하므로 PRADA와 같은 분포 기반 탐지 기법을 회피할 수 있음.

--------------------------------------------------------------------------------
논문 제공 알고리즘 (의사코드 요약)
논문은 별도의 의사코드 박스 대신 "The ACTIVETHIEF framework" 섹션-에서 단계별 절차를 서술하고 있습니다. 이를 요약하면 다음과 같습니다:
# Algorithm: ACTIVETHIEF Framework
Input: Thief Dataset (Unlabeled), Query Budget B, Initial Seed Size |S0|, Step Size k
Output: Substitute Model f_tilde

1. S_0 = Select_Random_Subset(Thief_Dataset, size=|S0|)
2. D_labeled = Query_Victim(f, S_0)  # 초기 쿼리

3. While (Current_Query_Count < B):
    # 대체 모델 학습 (매 라운드 초기화)
    f_tilde = Initialize_Model()
    Train(f_tilde, D_labeled) # Early stopping & F1 score selection
    
    # 후보군(Unlabeled Pool)에 대한 예측
    Unlabeled_Pool = Thief_Dataset - D_labeled.inputs
    Predictions = Predict(f_tilde, Unlabeled_Pool) # Softmax vector
    
    # 능동 학습 전략으로 다음 쿼리 선택
    S_next = Active_Learning_Strategy(Predictions, k)
             # Strategies: Random, Uncertainty, K-center, DFAL, etc.
    
    # 피해 모델 쿼리 및 데이터셋 업데이트
    D_new = Query_Victim(f, S_next)
    D_labeled = D_labeled + D_new

--------------------------------------------------------------------------------
추가 요청 정보 정리
1. 정보량 측정 방식 (Uncertainty Metric)
• 방식: 엔트로피(Entropy) 기반 불확실성 샘플링 (Uncertainty Sampling).
• 수식: 
H 
n
​
 =− 
j
∑
​
  
y
~
​
  
n,j
​
 log 
y
~
​
  
n,j
​
 
    ◦  
y
~
​
  
n,j
​
 : 대체 모델  
f
~
​
 가 샘플 x 
n
​
 에 대해 예측한 클래스 j의 확률.
    ◦ 엔트로피 H 
n
​
 이 가장 높은(가장 불확실한) 샘플을 선택함.
2. 후보 풀(Pool) 구성 및 갱신 규칙
• 구성: Thief 데이터셋(공개 데이터) 중 아직 쿼리하지 않은 나머지 샘플들로 구성 (D 
pool
​
 =X 
thief
​
 ∖∪S 
i
​
 ).
• 갱신:
    1. 대체 모델  
f
~
​
 를 사용하여 풀에 있는 모든 샘플에 대해 "근사 레이블(approximate labels)"과 "확률 벡터"를 계산.
    2. 선택 전략에 의해 k개가 선정되면, 이를 풀에서 제거하고 쿼리된 세트(Labeled Set)로 이동.
3. Selection Rule (선택 규칙)
• Top-k 방식: 각 전략별 점수(Score)를 계산하여 상위 k개를 선택.
    ◦ Uncertainty: 엔트로피 상위 k개.
    ◦ DFAL: 섭동 크기 α 
n
​
  하위 k개 (결정 경계에 가장 가까운 샘플).
    ◦ K-center: 현재 센터(Labeled set)로부터의 거리가 가장 먼 샘플을 하나씩 Greedy하게 k번 선택.
    ◦ DFAL + K-center: DFAL로 상위 ρ개(예: 전체 예산)를 먼저 필터링한 후, 그 중에서 K-center로 k개를 선택.
4. 라운드별 학습/Selection 순서
• 순서: Train → Predict Pool → Select → Query → Train (반복)
    ◦ 문헌의 Figure 2에 따르면:
        1. (초기) Random Selection → Query → Collect Labels.
        2. Train: 수집된 데이터로 모델 학습.
        3. Predict: 학습된 모델로 풀 데이터 예측.
        4. Select: 전략(Strategy)을 사용해 다음 쿼리셋 선정.
        5. Query: 선정된 샘플을 피해 모델에 쿼리하여 반복.
5. Augmentation 사용 여부 및 횟수
• 사용 여부: 논문의 실험 설정 섹션에서 데이터 증강(Augmentation)에 대한 명시적인 언급은 없습니다.
• 참고: DFAL 전략 설명 시 "perturbed sample  
x
^
  
n
​
 "을 생성하지만, 이는 선택 기준(metric)을 계산하기 위함이며, 훈련 데이터로 추가하거나 쿼리하는 용도가 아닙니다. 피해 모델에는 오직 원본 x 
n
​
 만 쿼리됩니다.
6. 기본 하이퍼파라미터 (상세)
• 학습 반복(Epochs): 최대 1,000 (Early Stopping: 이미지 100, 텍스트 20).
• Batch Size: 150 (이미지), 50 (텍스트).
• Learning Rate: Adam 기본값 사용 (논문에 수치 명시 안 됨, 통상 0.001).
• Dropout: 0.1 (CIFAR-10 제외), 0.2 (CIFAR-10).
• L2 Regularization: 0.001.
• Validation Split: 전체 쿼리 예산의 20%를 검증용으로 사용.
• Initial Seed Size: 전체 예산의 10%.