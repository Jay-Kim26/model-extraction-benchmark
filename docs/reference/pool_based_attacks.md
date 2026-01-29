1. Knockoff Nets (강화학습 기반 능동적 전이 학습)
• 핵심 아이디어: Victim 모델의 거동을 모방하기 위해, 어떤 이미지를 쿼리할지 결정하는 **정책(Policy)**을 강화학습으로 학습합니다.
• 구현 흐름:
    1. Transfer Set 선택: 공격자는 P 
A
​	
 (X) (예: ImageNet)에서 이미지를 선택.
    2. Adaptive Strategy (RL):
        ▪ 이미지 데이터셋의 계층적 구조(Hierarchy of labels)를 활용.
        ▪ Action: 특정 클래스(노드)를 선택하여 해당 클래스의 이미지를 샘플링.
        ▪ Reward (r 
t
​	
 ): 쿼리 결과가 Victim 모델의 **확신도(Certainty)**가 높거나, 기존 쿼리들과 **다양성(Diversity)**이 높거나, 현재 Clone 모델과 Loss 차이가 클수록 보상을 줌.
        ▪ Policy Update: Gradient Bandit 알고리즘을 사용하여 정책 π 업데이트.
    3. Training: 수집된 (x,y 
victim
​	
 ) 쌍으로 Knockoff 모델 학습 (Soft target Cross-Entropy).
• 구현 상세 (수식 & 파라미터):
    ◦ Reward Function: R=R 
cert
​	
 +R 
div
​	
 +R 
loss
​	
  (각 항목은로 스케일링).
        ▪ R 
cert
​	
 (y 
t
​	
 )=P(y 
t,k1
​	
 ∣x 
t
​	
 )−P(y 
t,k2
​	
 ∣x 
t
​	
 ) (Top-1과 Top-2 확률 차이).
        ▪ R 
div
​	
 : 이전 Δ 타임스텝 동안의 평균 예측값과의 차이.
    ◦ Policy Update: H 
t+1
​	
 (z 
t
​	
 )=H 
t
​	
 (z 
t
​	
 )+α(r 
t
​	
 − 
r
ˉ
  
t
​	
 )(1−π 
t
​	
 (z 
t
​	
 )).
    ◦ Architecture: Victim과 Knockoff 모두 ResNet-34 등을 사용하며, 공격자는 ImageNet 사전 학습된 모델로 시작.
2. InverseNet (데이터 복원 기반 공격)
• 핵심 아이디어: Victim 모델의 훈련 데이터를 역으로 복원(Inversion)하여, 원본 데이터와 유사한 분포의 합성 데이터를 생성해 학습합니다.
• 구현 흐름:
    1. Initialization: 공개 데이터셋에서 Coreset (k-Center Greedy) 알고리즘으로 대표 샘플을 뽑아 초기 Substitute 모델 학습.
    2. High Confidence Sample Selection (HCSS): 초기 모델을 이용해 결정 경계에서 먼(High Confidence) 샘플을 선별. 이를 위해 적대적 공격(DeepFool 등)을 수행했을 때 perturbation이 큰 샘플을 선택.
    3. Inversion Model (G 
V
​	
 ) 학습: Victim의 출력(Confidence vector)을 입력받아 이미지를 복원하도록 G 
V
​	
  학습.
        ▪ Loss: L(G 
V
​	
 (trunc 
1
​	
 (F 
V
​	
 (x))),x) (Truncation을 사용하여 Top-1 정보만으로 복원 유도).
    4. Retraining: G 
V
​	
 로 생성한 합성 데이터로 최종 Substitute 모델 재학습.
• 구현 상세:
    ◦ 쿼리 예산 분배 (K 
1
​	
 :K 
2
​	
 :K 
3
​	
 ): 초기화:역학습:재학습 비율 = 0.45 : 0.45 : 0.1.
    ◦ Inversion Model (G 
V
​	
 ): 5개의 Transposed Convolution 블록 (BatchNorm + Tanh, 마지막은 Sigmoid).
    ◦ Data Augmentation: 생성된 데이터에 cropping, scaling, rotating 등을 적용하여 데이터 수 증강.
3. SwiftThief (대조 학습 활용)
• 핵심 아이디어: 쿼리하지 않은 데이터(Unqueried Data)도 **대조 학습(Contrastive Learning)**을 통해 특징 추출에 활용하여 쿼리 효율성을 극대화합니다.
• 구현 흐름:
    1. 데이터셋 분할: 쿼리된 집합 Q와 쿼리되지 않은 집합 U.
    2. Sampling: 초기에는 Entropy 기반, 이후에는 희소 클래스 우선(Rarely Queried Class Prioritization) 전략 사용.
    3. Training Objective:
        ▪ L 
total
​	
 =L 
contrastive
​	
 +λ 
3
​	
 L 
matching
​	
 
• 구현 상세 (Loss Function):
    ◦ Self-Supervised Loss (U용): SimSiam 구조 사용. x의 두 가지 증강 버전에 대한 코사인 유사도 최대화.
    ◦ Soft-Supervised Loss (Q용): Victim의 Soft label을 활용한 대조 손실.
        ▪ Weight η 
ij
​	
 : 두 샘플 i,j의 Victim 출력 확률 분포가 유사할수록, 그리고 불확실성(Entropy)이 낮을수록 높은 가중치를 줌.
        ▪ η 
ij
​	
 =1[i

=j](1+ 
logK
H( 
y
^
​	
  
i
​	
 )
​	
 )(1+ 
logK
H( 
y
^
​	
  
j
​	
 )
​	
 )cos∠( 
y
^
​	
  
i
​	
 , 
y
^
​	
  
j
​	
 ).
    ◦ Regularizer: 소수 클래스에 대해 적대적 섭동(δ)을 주어도 표현이 흔들리지 않도록 하는 제약 추가.
    ◦ Hyperparameters: λ 
1
​	
 =1.0,λ 
2
​	
 =0.01. FGSM ϵ=0.01 (Regularizer용).
4. Black-box Dissector (Hard-Label 특화)
• 핵심 아이디어: Hard Label만 주는 Victim 환경에서 정보를 더 캐내기 위해 CAM 기반 지우기(Erasing) 전략과 Self-KD를 사용합니다.
• 구현 흐름:
    1. CAM-driven Erasing:
        ▪ Substitute 모델의 Grad-CAM을 계산하여 중요 영역을 파악.
        ▪ 해당 영역을 지운(Erased) 이미지를 Victim에 다시 쿼리. 만약 라벨이 바뀌면 "Top-2" 클래스 정보를 얻는 셈이고, 안 바뀌면 Substitute 모델의 Attention을 Victim과 맞추도록 학습.
    2. Self-KD with Random Erasing (RE):
        ▪ 데이터 부족 및 Hard label로 인한 과적합 방지.
        ▪ 입력 이미지를 N번 무작위로 지우고(Random Erasing), Substitute 모델에 입력하여 나온 출력의 평균을 Pseudo-label로 사용.
• 구현 상세:
    ◦ Erasing Selection: Grad-CAM 영역을 지웠을 때, Substitute 모델 기준 예측 확률 변화가 가장 큰 이미지를 선택하여 쿼리함 (정보 획득량 최대화).
    ◦ Final Loss: L=∑ 
D 
T
​	
 ∪D 
E
​	
 
​	
 L 
CE
​	
 (y 
victim
​	
 , 
y
^
​	
 )+∑ 
D 
P
​	
 
​	
 L 
CE
​	
 (y 
pseudo
​	
 , 
y
^
​	
 ).
5. CloudLeak (Adversarial Active Learning)
• 핵심 아이디어: 결정 경계(Decision Boundary) 근처의 샘플을 찾기 위해 FeatureFool이라는 적대적 공격 기법을 사용하여 합성 데이터를 생성합니다.
• 구현 흐름:
    1. FeatureFool: L-BFGS 최적화를 사용하여 다음을 만족하는 x 
′
  생성:
        ▪ Victim의 내부 특징(Feature) 표현을 크게 변화시키면서도(Triplet Loss 사용), 인간 눈에는 원본과 유사할 것.
        ▪ 최적화 식: mind(x 
′
 ,x)+λ⋅loss 
f,l
​	
 (x 
′
 )
        ▪ Triplet Loss: max(D(ϕ 
k
​	
 (x 
′
 ),ϕ 
k
​	
 (x 
t
​	
 ))−D(ϕ 
k
​	
 (x 
′
 ),ϕ 
k
​	
 (x 
s
​	
 ))+M,0).
    2. Active Learning: 생성된 샘플 중 Victim의 불확실성(Least Confidence)이 가장 높은 샘플을 쿼리 대상으로 선정.
    3. Transfer Learning: VGG19, ResNet50 등의 사전 학습 모델의 마지막 FC 레이어만 재학습(Fine-tuning).

--------------------------------------------------------------------------------
종합 구현 가이드라인 (Code Implementation Checklist)
1. Generator/Inverter 구조 통일:
    ◦ InverseNet과 같이 이미지를 생성해야 하는 경우, DCGAN의 Generator 구조(ConvTranspose + BatchNorm + ReLU)를 기본으로 하되, 마지막 레이어는 Tanh 또는 Sigmoid로 정규화된 이미지 출력.
2. Dataset Handling:
    ◦ Public Dataset Pool: ImageNet, CIFAR-100 등 대용량 데이터셋을 준비하여 P 
A
​	
 (X)로 사용.
    ◦ Augmentation: Black-box Dissector의 Random Erasing, InverseNet의 기하학적 변환 등을 torchvision.transforms 등으로 구현 필수.
3. Victim Model Interface:
    ◦ 입력: Tensor (Batch, C, H, W)
    ◦ 출력 모드 설정 가능하도록 구현: return_prob=True (Soft label), return_prob=False (Hard label/Argmax).
4. Hyperparameter Summary:
    ◦ SwiftThief: SimSiam Projection head 차원 (2048), Batch size (상대적으로 커야 함).
    ◦ InverseNet: Query Ratio (0.45 : 0.45 : 0.1).
    ◦ CloudLeak: Triplet Loss margin α=0.5.
5. Optimization:
    ◦ 대부분의 Substitute Model 학습에는 SGD with Momentum (0.9) 사용.
    ◦ Learning Rate는 0.1에서 시작하여 스케줄링(Decay) 적용.