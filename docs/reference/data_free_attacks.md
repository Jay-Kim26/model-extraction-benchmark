제공된 소스 파일들(DFME, DFMS, MAZE, GAME, ES Attack)을 바탕으로, **Data-Free Model Extraction/Stealing(데이터 없는 모델 추출/도용)** 공격들을 코드로 구현하기 위해 필요한 세부 요소들을 YAML 형식으로 정리했습니다.

이 문서들은 공통적으로 원본 데이터 없이(Data-Free) 피해 모델(Victim)을 복제하는 기술을 다루지만, 접근 방식(Hard/Soft Label, GAN 사용 여부, 최적화 방식 등)에 차이가 있습니다.

```yaml
Implementations:
  - Method: "DFME (Data-Free Model Extraction)"
    Source: "DFME.pdf"
    Algorithm_Flow:
      Type: "Generative Adversarial Training (Min-Max Game)"
      Steps:
        1: "Generator(G)가 랜덤 노이즈 z에서 쿼리 x 생성"
        2: "피해 모델(V)과 학생 모델(S)에 x를 입력하여 출력 획득"
        3: "Gradient Estimation: V는 블랙박스이므로 Forward Differences 방식을 사용하여 G에 대한 V의 경사(gradient)를 근사"
        4: "G 업데이트: V와 S 간의 불일치(Disagreement, L1 Loss)를 최대화하는 방향으로 학습 (Gradient Ascent)"
        5: "S 업데이트: V의 출력을 타겟으로 S를 학습 (L1 Loss 최소화)"
    Loss_Functions:
      Student_Loss: "L1 Norm Loss (Cross-Entropy나 KL-Divergence보다 수렴에 유리함)"
      Logit_Recovery: "Softmax 확률만 제공될 경우, 평균 보정(Mean Correction)을 통해 Logit 값을 근사 복원하여 Loss에 사용: l_i ≈ log(p_i) - mean(log(p))"
    Hyperparameters:
      Optimizer: "S는 SGD (LR 0.1), G는 Adam (LR 5e-4)"
      Iterations: "Generator steps (n_G) = 1, Student steps (n_S) = 5"
      Gradient_Est: "무작위 방향 수 m=1, 스텝 사이즈 epsilon=1e-3"
      Query_Budget: "SVHN: 2M, CIFAR-10: 20M"
    Architecture:
      Student: "ResNet-18-8x (Victim이 ResNet-34-8x일 때)"
      Generator: "3-layer CNN + Linear Upsampling + BatchNorm + ReLU + Tanh [Implementation assumption based on standard GANs, specific layers not detailed in excerpts]"

  - Method: "MAZE (Model Stealing via Zeroth-Order Gradient Estimation)"
    Source: "MAZE.pdf"
    Algorithm_Flow:
      Type: "Generative Model with Zeroth-Order Optimization"
      Steps:
        1: "Generator(G)가 노이즈 z ~ N(0, I)에서 입력 x 생성"
        2: "Clone(C)과 Target(T)의 출력 차이(KL-Divergence) 계산"
        3: "G 업데이트: L_G = -D_KL(T(x) || C(x))를 최대화 (Target과 다르게 예측하도록 유도하여 탐색 공간 확장)"
        4: "C 업데이트: L_C = D_KL(T(x) || C(x))를 최소화 (Target을 모방)"
        5: "Zeroth-Order Gradient: G 업데이트 시 T의 미분값을 모르므로 Forward Differences로 근사"
    Hyperparameters:
      Batch_Size: 128
      Steps_Per_Iter: "N_G=1 (Generator), N_C=5 (Clone), N_R=10 (Experience Replay)"
      Queries_Per_Iter: "B * (N_G * (m + 1) + N_C) (여기서 m은 gradient estimation directions)"
      Query_Budget: "FashionMNIST/SVHN: 5M, CIFAR-10/GTSRB: 30M"
    Architecture:
      Clone_Model: "22-layer WideResNet (Target 지식 없음 가정)"

  - Method: "DFMS-HL (Data-Free Model Stealing - Hard Label)"
    Source: "DFMS.pdf"
    Algorithm_Flow:
      Type: "GAN-based with Hard Labels"
      Context: "피해 모델이 확률(Soft Label)이 아닌 Top-1 클래스(Hard Label)만 반환하는 경우"
      Steps:
        1: "Generator(G) 초기화: Proxy 데이터나 합성 데이터를 사용하여 DCGAN 사전 학습"
        2: "Clone(C)과 G를 교대로 학습"
        3: "G 업데이트: Adversarial Loss + Diversity Loss 사용. Discriminator는 가짜와 진짜 Proxy 데이터를 구별"
        4: "C 업데이트: G가 생성한 데이터와 V의 Hard Label 쌍(x, y_hat)을 사용하여 학습"
    Loss_Functions:
      Diversity_Loss: "생성된 이미지의 다양성을 확보하기 위해 Clone 모델의 예측 엔트로피를 최대화하거나 특정 분포를 따르도록 유도 (Clone을 Proxy로 사용)"
    Query_Budget: "CIFAR-10 기준 800만(8M) 쿼리 (소프트 라벨 방식은 20M 필요)"

  - Method: "GAME (Generative-Based Adaptive Model Extraction)"
    Source: "GAME.pdf"
    Algorithm_Flow:
      Type: "AC-GAN + Active Learning"
      Core_Idea: "결정 경계(Decision Boundary) 근처의 샘플을 생성하되 타겟 분포의 중심에서 벗어나지 않도록 함"
      Phases:
        1: "Target Distribution Learning (TDL): Proxy 데이터로 AC-GAN 학습"
        2: "Active Categories Selecting (ACS): 학습 효율이 높은 카테고리를 능동적으로 선택"
        3: "Adaptive Generator Updating (AGU): Substitute 모델(N_S)을 Shadow 모델로 사용하여 G를 미세 조정"
        4: "Generative Model Distillation (GMD): 생성된 샘플로 N_S 학습"
    Loss_Functions:
      Generator_Total_Loss: "beta1*L_res + beta2*L_bou + beta3*L_adv + beta4*L_dif"
      Components:
        L_res: "Model Responsivity (N_S의 logits 출력 크기 증가)"
        L_bou: "Boundary Support (결정 경계 근처 샘플 생성)"
        L_adv: "Adversarial (N_S를 속이는 샘플)"
        L_dif: "Prediction Difference (N_S와 N_V의 불일치 최대화 - KL Divergence)"

  - Method: "ES Attack (Estimate & Synthesize)"
    Source: "esattack.pdf"
    Algorithm_Flow:
      Type: "Iterative Substitution"
      Steps:
        1: "랜덤 합성 데이터 D(0)_syn으로 시작"
        2: "E-Step (Estimate): 합성 데이터를 Victim에 쿼리하여 라벨링 후 Substitute Model(f_s) 학습 (Knowledge Distillation)"
        3: "S-Step (Synthesize): 학습된 f_s의 정보를 바탕으로 더 나은 합성 데이터 D(t)_syn 생성"
        4: "위 과정을 반복하여 f_s의 정확도 향상"
    Data_Synthesis_Strategies:
      DNN-SYN: "Generator 네트워크(G)를 사용하여 데이터 생성. G와 f_s를 교대로 학습"
      OPT-SYN: "최적화 기반 방법. 노이즈 이미지 x를 f_s의 특정 클래스 확률을 최대화하도록 업데이트 (Adam 사용)"
    Hyperparameters:
      OPT-SYN: "매 epoch마다 30 iteration 데이터 합성, LR=0.01. Substitute 모델 10 epoch 학습."
      DNN-SYN: "LR=0.001, G와 f_s를 교대로 학습."

Ambiguities_and_Notes:
  - Generator_Architecture: "대부분의 논문(DFME, MAZE 등)에서 Generator의 정확한 레이어 구성(커널 크기 등)보다는 'DCGAN 기반' 또는 'ResNet 기반'과 같이 언급하므로, 표준적인 GAN 구조를 가정하고 입출력 차원을 데이터셋(CIFAR-10 등)에 맞춰 구현해야 함."
  - Query_Accounting: "Gradient Estimation(DFME, MAZE)을 사용하는 경우, 1번의 G 업데이트를 위해 m+1번의 쿼리가 소모됨을 명심해야 함 (m은 방향 벡터 수). 구현 시 쿼리 예산 초과 방지 로직 필수."
  - Logit_Recovery_Implementation: "DFME와 MAZE는 Victim의 출력(Softmax)에서 Logit을 복원하는 과정이 성능에 치명적임. Mean Correction 수식 구현이 필수적임."
```

추가정보
제공된 문헌(DFME, DFMS, GAME, ES Attack)을 바탕으로 요청하신 세부 구현 정보를 정리해 드립니다.

### 1. 입력/출력 스케일 및 정규화 방식 (Input/Output Scale & Normalization)

대부분의 Data-Free 공격 방법론은 생성 모델(Generator)을 사용하여 이미지를 합성하므로, 생성된 이미지의 값 범위(Range)와 피해 모델(Victim Model)이 기대하는 입력 범위의 일치가 중요합니다.

*   **DFME (Data-Free Model Extraction):**
    *   **스케일:** Generator의 마지막 레이어에서 **Tanh** 활성화 함수를 사용하여 출력 값을 **$[-1, 1]$** 범위로 제한합니다.
    *   **구현:** 피해 모델(Oracle)이 $[-1, 1]$ 범위의 입력을 받는다고 가정하거나, 이 범위의 출력을 피해 모델의 전처리 방식(예: ImageNet 정규화)에 맞게 변환해야 합니다. Gradient Approximation 시에는 Tanh 적용 직전의 값(pre-activation images)을 사용하여 경계값 문제를 피합니다.
*   **DFMS (Data-Free Model Stealing - Hard Label):**
    *   **스케일:** DCGAN 기반 Generator의 마지막 레이어에서 **Tanh**를 사용하여 **$[-1, 1]$** 범위의 이미지를 생성합니다.
*   **ES Attack (Estimate & Synthesize):**
    *   **스케일:** DNN-SYN 방식의 Generator는 마지막 레이어에 **Tanh** 함수를 사용하여 **$(-1, 1)$** 범위 내의 합성 이미지를 출력합니다.

### 2. Logit Recovery의 정확한 수식 (DFME)

피해 모델이 확률 벡터(Softmax output, $p$)만 반환하고 로짓(Logit, $v$)을 제공하지 않을 때, L1 손실 함수 등을 계산하기 위해 로짓을 복원하는 과정입니다.

*   **방식:** Mean Correction (평균 보정).
*   **수식:**
    $$l_i \approx \log(p_i) - \frac{1}{K}\sum_{j=1}^{K} \log(p_j)$$
    *   여기서 $p_i$는 클래스 $i$의 확률, $K$는 전체 클래스 개수입니다.
    *   **원리:** Softmax는 이동 불변성(shift-invariant)을 가지므로 ($Softmax(v) = Softmax(v+c)$), 실제 로짓의 평균이 0에 가깝다고 가정하고 로그 확률(log-probabilities)에서 그 평균을 빼줌으로써 중심화된 로짓(centered logits)을 근사합니다.

### 3. GAME의 복합 Loss 정확한 수식

GAME은 생성자(Generator)를 업데이트하기 위해 4가지 서로 다른 목적을 가진 손실 함수의 가중 합을 사용합니다.

*   **전체 손실 함수 (Total Loss):**
    $$Loss_{Total} = \beta_1 \cdot L_{res} + \beta_2 \cdot L_{bou} + \beta_3 \cdot L_{adv} + \beta_4 \cdot L_{dif}$$
    *   $\beta_i$: 각 손실 항의 균형을 맞추기 위한 가중치 계수.

*   **세부 구성 요소:**
    1.  **Model Responsivity ($L_{res}$):** 학생 모델(Substitute, $N_S$)의 로짓(logits, $f_S$) 출력 크기를 키워 주요 특징을 활성화하도록 유도합니다.
        $$L_{res} = - \sum_{i=1}^{N} \max(0, f_S^i)$$
        (여기서 $f_S^i$는 학생 모델의 $i$번째 클래스 로짓값, $N$은 배치 내 샘플 수 혹은 클래스 수로 문맥상 해석됨).
    2.  **Boundary Support ($L_{bou}$):** 결정 경계(Decision Boundary) 근처의 샘플을 생성하기 위해 가장 높은 확률과 두 번째로 높은 확률의 차이를 최소화합니다.
        $$L_{bou} = N_S(x)_{top1} - N_S(x)_{top2}$$
       .
    3.  **Adversarial Correction ($L_{adv}$):** 생성된 샘플이 학생 모델을 속이도록(adversarial) 유도하여 결정 경계를 넘나들게 합니다.
        $$L_{adv} = -CE(N_S(x), \arg\max_i N_S(x)_i)$$
        (자신의 예측 레이블에 대한 Cross Entropy를 음수화하여, 해당 예측을 하지 않도록 유도).
    4.  **Prediction Difference ($L_{dif}$):** 학생 모델($N_S$)과 피해 모델($N_V$) 간의 예측 불일치를 최대화합니다 (Disagreement Maximization).
        $$L_{dif} = -KL(N_S(x), N_V(x))$$
        (KL Divergence의 음수값).

### 4. ES Attack의 합성 이미지 초기화 및 정규화

ES Attack은 초기화된 랜덤 데이터로부터 반복적으로 데이터를 정제해 나가는 방식을 취합니다.

*   **초기화 (Initialization):**
    *   알고리즘의 시작 시점($t=0$)에서 합성 데이터셋 $D_{syn}^{(0)}$은 **가우시안 분포(Gaussian Distribution, $\mathcal{N}(0, 1)$)** 에서 무작위로 샘플링하여 초기화합니다.
    *   OPT-SYN 방식의 경우, 각 데이터 샘플 $x$를 가우시안 분포로부터 초기화한 후 최적화를 수행합니다.
*   **정규화/출력 범위:**
    *   DNN-SYN 방식의 Generator는 마지막 레이어에 **Tanh**를 사용하여 출력을 **$(-1, 1)$** 사이로 제한합니다. 구현 시 이 범위를 피해 모델이 요구하는 입력 범위(예: $$ 또는 ImageNet 정규화 범위)로 매핑해야 합니다.


### MAZE Replay Buffer

1. Replay Buffer (데이터셋 D) 구성 및 크기
• 구성: Clone Training 단계에서 생성기(Generator)가 만든 합성 입력 x와 이에 대한 타겟 모델의 예측값 T(x) 쌍인 (x,T(x))가 데이터셋 D에 저장됩니다.
• 크기: 제공된 텍스트에는 버퍼의 최대 크기(Capacity)에 대한 명시적인 제한(예: FIFO 큐 크기)은 언급되어 있지 않습니다. 대신, 각 라운드에서 생성된 데이터가 D에 누적(D ← D ∪ {(x, T(x))})되는 방식으로 기술되어 있어, 학습이 진행됨에 따라 데이터셋의 크기가 증가하는 구조임을 알 수 있습니다.
2. 샘플링 방식 (Sampling)
• 방식: 알고리즘 1에 기술된 Experience Replay 단계에서 (x, y_T) ∼ D로 표기되어 있습니다. 이는 저장된 데이터셋 D로부터 **무작위 샘플링(Random Sampling)**을 수행함을 의미합니다.
• 목적: 이전에 보았던 쿼리들에 대해 클론 모델을 재학습시킴으로써, 새로운 데이터를 학습할 때 이전 지식을 잊어버리는 Catastrophic Forgetting 문제를 방지하기 위함입니다.
3. 관련 하이퍼파라미터
• N 
R
​	
  (Iterations): Experience Replay 단계에서 수행하는 반복 횟수입니다. 실험 설정에서는 기본값으로 **N 
R
​	
 =10**을 사용했습니다.
• 배치 처리: Replay 단계에서도 별도의 쿼리 비용은 발생하지 않으며(이미 저장된 데이터 사용), 기존 학습 루프와 동일한 배치 크기(Batch Size)를 가정할 수 있습니다. 실험에서는 기본 배치 크기 B=128을 사용했습니다.

### 입력 전처리 방식

. 입력 차원 및 형식 (Input Dimensions & Formats)
피해 모델은 특정 데이터셋에 맞춰 훈련되었으므로, 공격자는 생성한 쿼리(Query)를 해당 차원에 맞춰야 합니다.
• GAME:
    ◦ 피해 모델의 입력 요구사항에 맞추기 위해 모든 이미지를 32x32 크기로 리사이즈(resize)하여 사용합니다.
    ◦ 실험에서 MNIST는 원본 데이터셋, Fashion-MNIST는 프록시 데이터셋으로 사용되었으며, 두 경우 모두 모델 입력에 맞게 조정됩니다.
• ES Attack:
    ◦ 피해 모델이 훈련된 데이터셋의 원본 해상도를 따릅니다.
    ◦ MNIST/KMNIST: 28x28 크기의 그레이스케일(Grey) 이미지.
    ◦ SVHN/CIFAR-10: 32x32 크기의 RGB 이미지.
• MAZE:
    ◦ 생성자(Generator) 모델 설계 시, 활성화 함수 결과를 업샘플링(upsampling)하여 공격 대상 데이터셋에 해당하는 **정확한 차원(correct dimensionality)**의 출력을 생성하도록 보장합니다.
    ◦ 생성자는 잠재 벡터(latent vector) z를 받아 타겟 분류기의 입력 차원과 일치하는 x∈R 
d
 를 생성합니다.
2. 생성자 출력 제어 (Generator Output Control)
공격자는 피해 모델이 처리할 수 있는 유효한 이미지 범위(Valid Image Range) 내의 데이터를 생성하기 위해 생성자의 마지막 레이어를 조정합니다. 이는 피해 모델의 입력 전처리 요구사항을 역으로 만족시키기 위함입니다.
• DFME:
    ◦ 생성자(Generator)의 마지막 레이어에서 Tanh 활성화 함수를 사용하여 출력 값을 [−1,1] 범위로 제한합니다. 이는 피해 모델이 해당 범위의 입력을 수용하거나, 이 범위의 값이 피해 모델의 입력 파이프라인에서 적절히 처리된다고 가정함을 의미합니다.
• ES Attack:
    ◦ DNN-SYN 방식의 생성자 역시 Tanh를 사용하여 출력을 생성하며, OPT-SYN 방식은 최적화 과정을 통해 데이터를 생성합니다.
    ◦ 합성된 데이터의 품질을 높이기 위해 랜덤 수평 뒤집기(horizontal flip), 수평 이동(horizontal shift), 가우시안 노이즈 추가 등의 증강(augmentation) 기법을 적용하여 피해 모델에 입력합니다.
3. 블랙박스 가정에 따른 전처리 (Black-box Assumption)
대부분의 문헌에서 피해 모델은 **블랙박스(Black-box)**로 취급되므로, 공격자는 피해 모델 내부에서 수행되는 구체적인 정규화(예: ImageNet Mean/Std subtraction) 과정을 알지 못한다고 가정합니다. 대신 공격자는 다음을 수행합니다.
• 입력 공간 탐색: 생성자를 통해 피해 모델이 학습된 데이터 분포(Target Distribution)와 유사한 입력을 생성하도록 학습합니다,.
• 쿼리 호환성: 생성된 합성 데이터가 피해 모델의 API가 허용하는 텐서(Tensor) 형태와 값의 범위를 갖추도록 합니다.
요약하자면, 문헌상에서 피해 모델의 전처리 방식은 **"데이터셋 고유의 해상도(예: 32x32 RGB) 준수"**와 **"생성자 출력을 통한 유효 픽셀 범위(예: Tanh를 통한 [-1, 1]) 매핑"**으로 정의됩니다.

### Generator/Discriminator Architechture

대부분의 Data-Free 공격은 표준적인 DCGAN 또는 AC-GAN 구조를 변형하여 사용하며, 일부 방법론(DFME, MAZE)은 별도의 Discriminator 네트워크 대신 피해 모델(Victim)과 대체 모델(Student/Clone)의 불일치를 이용합니다.
1. DFME (Data-Free Model Extraction)
DFME는 별도의 Discriminator 네트워크를 두지 않고, Victim과 Student 모델 간의 불일치(L1 Loss)를 최대화하는 방식으로 Generator를 학습시킵니다.
• Generator 아키텍처:
    ◦ 구조: 3개의 합성곱 레이어(Convolutional layers)로 구성됩니다.
    ◦ 구성 요소: 각 합성곱 레이어 사이에는 Linear Up-sampling 레이어, Batch Normalization 레이어, ReLU 활성화 함수가 삽입되어 있습니다.
    ◦ 출력층: 마지막 레이어에는 Tanh (Hyperbolic Tangent) 활성화 함수를 사용하여 출력 값을 [−1,1] 범위로 제한합니다.
    ◦ 초기화: 입력 z는 표준 정규 분포 N(0,1)에서 샘플링됩니다.
2. DFMS-HL (Data-Free Model Stealing - Hard Label)
DFMS는 DCGAN (Deep Convolutional GAN) 아키텍처를 기반으로 하며, 명시적인 Discriminator를 사용하여 진짜 Proxy 데이터와 생성된 데이터를 구별합니다.
• Generator 아키텍처:
    ◦ 구조: DCGAN 기반으로, 최대 5개의 Transpose Convolution (전치 합성곱) 레이어를 사용합니다.
    ◦ 구성 요소: 각 레이어 뒤에는 Batch Normalization과 ReLU 유닛이 따릅니다.
    ◦ 출력층: 마지막 합성곱 레이어 뒤에 Tanh 활성화 유닛을 사용하여 이미지를 정규화된 범위 [−1,1]로 생성합니다.
• Discriminator 아키텍처:
    ◦ 구조: 5개의 Convolution 레이어 스택으로 구성됩니다.
    ◦ 구성 요소: 각 레이어 뒤에는 Batch Normalization과 Leaky ReLU 유닛이 따릅니다.
    ◦ 출력층: 마지막 레이어는 Sigmoid 활성화 함수를 사용합니다.
3. GAME (Generative-Based Adaptive Model Extraction)
GAME은 AC-GAN을 기반으로 하며, Generator와 Discriminator 모두 구체적인 레이어 구성이 명시되어 있습니다.
• Generator 아키텍처:
    ◦ 구조: 3개의 Convolutional 레이어로 구성됩니다.
    ◦ 구성 요소: 각 레이어 사이에는 Up-sampling 레이어, Batch Normalization 레이어, ReLU 활성화 함수가 인터리브(interleaved) 방식으로 배치됩니다.
• Discriminator 아키텍처:
    ◦ 구조: 5개의 Convolutional 레이어로 구성됩니다.
    ◦ 구성 요소: 각 레이어 사이에는 ReLU 활성화 함수, Dropout 레이어(확률 0.25), Batch Normalization 레이어가 배치됩니다.
    ◦ 특이사항: 마지막 레이어에는 Batch Normalization을 적용하지 않습니다.
4. ES Attack (DNN-SYN 방식)
ES Attack의 DNN-SYN 방식은 ACGAN의 주요 설계를 따르며, 레이블 조건부(label conditioning) 생성을 수행합니다.
• Generator 아키텍처:
    ◦ 입력: 랜덤 잠재 벡터(latent vector) z와 원-핫(one-hot) 레이블 벡터 l을 연결(concatenate)하여 입력으로 사용합니다.
    ◦ 구조: 4개의 Transposed Convolution 레이어를 사용하여 업샘플링을 수행합니다.
    ◦ 구성 요소: 각 Transposed Convolution 레이어 뒤에는 Batch Normalization 레이어와 ReLU 활성화 함수가 따릅니다 (마지막 레이어 제외).
    ◦ 출력층: 마지막 레이어에서는 Tanh 함수를 사용하여 (−1,1) 범위의 합성 이미지를 출력합니다.
5. MAZE (Model Stealing via Zeroth-Order Gradient Estimation)
MAZE는 DFME와 유사하게 별도의 Discriminator 없이 Generator를 학습시키며, 데이터셋에 따라 출력 차원을 맞춥니다.
• Generator 아키텍처:
    ◦ 구조: 3개의 Convolutional 레이어로 구성된 생성 모델을 사용합니다.
    ◦ 구성 요소: G의 각 합성곱 레이어 뒤에는 Batch Normalization 레이어가 따르며, 활성화(activations)는 공격 대상 데이터셋에 맞는 차원을 확보하기 위해 Up-sampled 됩니다.
요약 및 구현 팁
• 공통점: 대부분의 Generator는 3~5개의 Conv/TransposeConv 레이어 + Batch Norm + ReLU 구조를 가지며, 마지막에 Tanh를 사용하여 출력을 스케일링합니다.
• Discriminator 유무: DFMS와 GAME은 명시적인 Discriminator 네트워크(Conv + Leaky ReLU/ReLU)를 구현해야 하며, DFME와 MAZE, ES Attack은 Discriminator 대신 Victim 및 Substitute 모델의 출력을 Loss 계산에 활용하므로 별도의 Discriminator 네트워크 구현이 필요 없습니다.

### GAME ACS 세부 구현

1. ACS의 목적 및 작동 방식
AC-GAN의 생성자 G는 잠재 노이즈 z와 타겟 레이블 y 
g
​	
 를 입력받아 샘플을 생성합니다. ACS 단계는 이 y 
g
​	
 를 결정하는 확률 분포 P 
π
​	
 를 계산하고 업데이트합니다.
• 프로세스:
    1. 현재 대체 모델(N 
S
​	
 )의 상태를 기반으로 각 클래스에 대한 선택 확률(P 
π
​	
 )을 계산합니다.
    2. 이 확률 분포에 따라 생성할 샘플의 레이블 Y 
g
​	
 를 샘플링합니다 (Y 
g
​	
 ∼P 
π
​	
 ).
    3. 생성자 G는 Y 
g
​	
 를 입력받아 해당 클래스의 이미지를 생성하여 피해 모델에 쿼리합니다.
2. 세부 선택 전략 (Strategies)
문헌에서는 두 가지 구체적인 전략을 제안하고 있습니다.
A. 예측 불확실성 (Prediction Uncertainty)
대체 모델(N 
S
​	
 )이 가장 확신하지 못하는(불확실한) 클래스를 우선적으로 선택하는 전략입니다. 대체 모델이 해당 클래스에 대해 낮은 신뢰도를 보인다면, 그 클래스에 대한 추가적인 정보(학습 데이터)가 필요하다는 가정에 기반합니다.
• 불확실성 점수 (c 
i
​	
 ): 생성된 샘플에 대한 대체 모델의 소프트맥스(softmax) 출력 중 최대값을 1에서 뺀 값으로 정의합니다. 
c 
i
​	
 =1−max{softmax[N 
S
​	
 (G(z,i))]}
• 선택 확률 (P 
i
unc
​	
 ): 각 클래스의 불확실성 점수를 전체 합으로 나누어 정규화합니다. 
P 
i
unc
​	
 = 
∑ 
t=1
N
​	
 c 
t
​	
 
c 
i
​	
 
​	
 
 여기서 N은 전체 클래스 개수입니다.
B. 편차 거리 (Deviation Distance)
대체 모델(N 
S
​	
 )과 피해 모델(N 
V
​	
 ) 간의 예측 불일치가 가장 큰 클래스를 선택하는 전략입니다. 두 모델의 예측이 다르다는 것은 대체 모델이 아직 피해 모델을 충분히 모방하지 못했음을 의미하므로, 해당 클래스를 집중적으로 학습합니다.
• 거리 점수 (d 
i
​	
 ): 동일한 생성 샘플 G(z,i)에 대해 대체 모델과 피해 모델(블랙박스 API, N 
V
∗
​	
 )의 출력 간 **쿨백-라이블러 발산(KL-Divergence)**을 계산합니다. 
d 
i
​	
 =KL[N 
S
​	
 (G(z,i)),N 
V
∗
​	
 (G(z,i))]
• 선택 확률 (P 
i
dev
​	
 ): 거리 점수를 정규화하여 확률로 사용합니다. 
P 
i
dev
​	
 = 
∑ 
t=1
N
​	
 d 
t
​	
 
d 
i
​	
 
​	
 
 .
3. 성능 비교 및 권장 전략
실험 결과(Ablation Study)에 따르면, 예측 불확실성(Uncertainty-based) 전략이 무작위(Random) 선택이나 편차 거리(Deviation-based) 전략보다 전반적으로 더 우수한 성능(Fidelity 및 Accuracy)을 보였습니다.
• 특히 쿼리 예산이 증가함에 따라 불확실성 기반 전략의 성능 향상 폭이 가장 컸으며, 문헌에서는 이를 **GAME 공격을 수행할 때 가장 적합한 전략(best practice)**으로 결론지었습니다.