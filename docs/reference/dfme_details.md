1. DFME 구현 정보 요약표
항목
상세 내용 및 출처
1. 알고리즘 절차
Generator(G)와 Student(S)의 적대적 학습 (Min-Max Game):<br>1. G가 노이즈 z로부터 쿼리 x 생성.<br>2. S는 V의 출력을 모방하도록 학습 (Disagreement 최소화).<br>3. G는 S와 V의 출력이 달라지도록 학습 (Disagreement 최대화).<br>4. V의 미분값 부재를 해결하기 위해 Zeroth-order gradient estimation 사용.
2. 쿼리 생성/선택 방식
합성 쿼리 생성 (Generative):<br>- 정의: x=G(z) (z는 랜덤 노이즈).<br>- 기준: Student와 Victim 간의 불일치(Disagreement)를 최대화하는 x를 생성.<br>- 제약: 출력은 Tanh를 통해 [−1,1] 범위로 제한.
3. 학습 루프 구조
Online Iterative Training:<br>Query Budget Q가 소진될 때까지 G-step과 S-step을 교차 반복.<br>Query → Train G → Query → Train S 구조 (배치 단위 반복).
4. 하이퍼파라미터
- Batch Size: 256.<br>- Optimizer: S는 SGD (lr 0.1), G는 Adam (lr 5e-4).<br>- Steps: n 
G
​
 =1 (Generator), n 
S
​
 =5 (Student).<br>- Gradient Approx: m=1 (random directions), ϵ=10 
−3
 .
5. 손실 함수 구성
L1 Norm Loss:<br>$L(x) = \sum_{i=1}^{K}
6. 데이터/전처리
Data-Free: 외부 데이터 없음.<br>Input: 표준 정규분포 z∼N(0,1).<br>Logit Recovery: Victim의 Softmax 확률 출력에서 평균을 빼서 Logit을 근사함 (Mean Correction).
7. 모델 아키텍처
Student: ResNet-18-8x.<br>Generator: 3 Conv layers + Linear Upsampling + BatchNorm + ReLU + Tanh.<br>Victim: ResNet-34-8x (실험 기준).
8. 구현 주의사항
- KL Divergence 사용 금지: 학습 후반부에 Gradient Vanishing 문제 발생.<br>- Gradient Approximation: m=1(부정확한 미분)이어도 충분하며 쿼리 효율이 더 중요함.<br>- Victim Output: 단순 확률(Probability)을 그대로 쓰지 말고 반드시 Logit으로 변환해야 함.

--------------------------------------------------------------------------------
2. 논문 제공 알고리즘 (의사코드 요약)
논문의 Algorithm 1: Data-Free Model Extraction을 요약한 내용입니다.
# Algorithm 1: DFME
Input: Query budget Q, Generator iters n_G, Student iters n_S, 
       Learning rate eta, random directions m, step size epsilon
Result: Trained Student S

While Q > 0 do:
    # 1. Generator Update (Disagreement Maximization)
    For i = 1 to n_G do:
        z ~ Normal(0, 1)
        x = G(z)
        # Victim(V)은 블랙박스이므로 Forward Differences로 G의 gradient 근사
        grad_G = Approximate_Gradient(V, S, x, m, epsilon) 
        # Gradient Ascent (Disagreement 증가 방향)
        theta_G = theta_G + eta * grad_G 
    End

    # 2. Student Update (Disagreement Minimization)
    For i = 1 to n_S do:
        z ~ Normal(0, 1)
        x = G(z)
        # V와 S의 출력(Logit) 계산
        loss = L1_Loss(V(x), S(x))
        grad_S = Calculate_Gradient(loss, theta_S)
        # Gradient Descent (Disagreement 감소 방향)
        theta_S = theta_S - eta * grad_S
    End

    Update Q (consumed queries)
End

--------------------------------------------------------------------------------
3. 추가 요청 정보 정리
Generator/Student/Victim 업데이트 순서와 스케줄
• 순서: Generator Update (n 
G
​
 회 반복) → Student Update (n 
S
​
 회 반복) → 반복.
• 스케줄: 전체 쿼리 예산 Q가 0이 될 때까지 무한 루프 (Epoch 개념이 아닌 쿼리 예산 기준).
Loss 구성 (정확한 수식, 각 항의 의미)
• 수식: 
L 
ℓ1
​
 (x)= 
i=1
∑
K
​
 ∣v 
i
​
 −s 
i
​
 ∣
• 구성 요소:
    ◦ v 
i
​
 : Victim 모델의 i번째 클래스에 대한 Logit (근사값).
    ◦ s 
i
​
 : Student 모델의 i번째 클래스에 대한 Logit.
    ◦ 의미: 두 모델의 Logit 값 간의 절대 차이 합(Mean Absolute Error). KL Divergence는 Student가 Victim에 수렴할수록 기울기가 소실(vanishing)되므로 L1 Loss가 수렴성과 안정성에 더 유리함.
• Logit Recovery: Victim이 확률 p만 줄 경우, l 
i
​
 ≈log(p 
i
​
 )−mean(log(p)) 공식을 사용해 Logit을 복원하여 Loss 계산에 사용.
입력 노이즈 분포/샘플링 방식
• 분포: 표준 정규 분포 (Standard Normal Distribution).
• 방식: z∼N(0,1).
G-step/S-step 비율, Iteration 수
• 비율: n 
G
​
 :n 
S
​
 =1:5.
• 이유: Generator 학습 시 Gradient Approximation 비용(쿼리)이 발생하므로, G를 적게 업데이트하고 S를 많이 업데이트하는 것이 쿼리 효율적임.
• Gradient Approx Iteration: m=1 (Forward Difference 방향 수). 논문은 m=10과 m=1의 성능 차이가 크지 않음을 확인.
사용한 모델 아키텍처 기본값
• Student: ResNet-18-8x (ResNet-18보다 채널이 8배 넓은 모델일 가능성이 높음, 논문 표기는 ResNet-18-8x).
• Generator:
    ◦ 3개의 Convolutional Layer.
    ◦ 각 Conv 층 사이에 Linear Up-sampling 배치.
    ◦ Batch Normalization 및 ReLU 사용 (마지막 층 제외).
    ◦ 마지막 활성화 함수: Hyperbolic Tangent (Tanh) → 출력 범위 [−1,1].
• Victim: 실험에서는 ResNet-34-8x 사용.
기본 하이퍼파라미터
• Query Budget (Q): SVHN 2백만(2M), CIFAR-10 2천만(20M).
• Student Optimizer: SGD, LR=0.1, Weight Decay=5e-4, Momentum 정보는 명시되지 않았으나 통상값 추정. LR Scheduler는 0.1, 0.3, 0.5 지점(전체 학습 진행도 기준)에서 0.3배 감쇠.
• Generator Optimizer: Adam, LR=5e-4. LR Scheduler는 10%, 30%, 50% 지점에서 0.3배 감쇠.
• Gradient Approximation: Step size ϵ=10 
−3
 , Directions m=1.
• Batch Size: 256.
L1 손실 함수가 KL 발산보다 추출 성능이 뛰어난 구체적인 이유는 무엇인가요?
블랙박스 모델의 확률 출력값에서 로짓을 복원하는 Mean Correction 수식을 알려주세요.
쿼리 예산을 줄이기 위해 제안된 하이브리드 추출 전략은 어떤 방식으로 작동하나요?