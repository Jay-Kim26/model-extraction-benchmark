# TODO - Model Extraction Benchmark Implementation

## Completed (Before Jan 22, 2026)
- Track B: DFME가 substitute를 state에 저장하도록 수정
- Track A KL NaN 이슈 해결 및 전체 테스트 통과
- DFME 기본 단위 테스트 추가
- Config validation 테스트 추가 및 준비
- 통합 테스트(간단) 정상 실행 확인
- Track B: missing substitute 시 metric을 숫자(0.0)로 일관화
- DFME 축소 budget end-to-end 테스트 추가
- ActiveThief selection 로직(uncertainty/k-center/dfal) 테스트 추가
- Cache cleanup 옵션(`delete_on_finish`) 테스트 추가
- Artifact 스키마(summary.json/metrics.csv) 테스트 추가

---

## Completed (Jan 22, 2026) - New Attack Implementation

### Phase 1: Attack Implementations (Files Created)
- [x] SwiftThief (`mebench/attackers/swiftthief.py`)
- [x] Black-box Dissector (`mebench/attackers/blackbox_dissector.py`)
- [x] CloudLeak (`mebench/attackers/cloudleak.py`)

### Phase 2: Configuration Files (Files Created)
- [x] `configs/swiftthief_experiment.yaml`
- [x] `configs/dissector_experiment.yaml`
- [x] `configs/cloudleak_experiment.yaml`

### Phase 3: Integration
- [x] Register attacks in `mebench/core/engine.py`
- [x] Export in `mebench/attackers/__init__.py`
- [x] Update `mebench/core/validate.py` output mode constraints

### Phase 4: Testing
- [x] Config validation tests for new attacks (`tests/test_new_attacks_config.py`)
- [x] All 7 tests passed

---

## Open Issues & Improvements (검토 결과 기반)

### 1. SwiftThief (`mebench/attackers/swiftthief.py`)

**문제점:**
- (stale) 이전 TODO에 placeholder/pass 및 scope 이슈가 있었으나, 현재 코드 기준으로는 `pass` placeholder가 보이지 않음(재검증 필요)

**개선사항:**
- [ ] (fidelity) 논문에서 정의된 soft-supervised contrastive / sharpness term 구현이 논문과 동일한지 재검증
- [ ] (fidelity) Augmentation이 논문/공식 구현과 동일한지 재검증

---

### 2. Black-box Dissector (`mebench/attackers/blackbox_dissector.py`)

**문제점:**
- (stale) 이전 TODO에 pseudo-label loss placeholder가 있었으나, 현재 코드 기준으로는 명시적 `pass` placeholder가 보이지 않음(재검증 필요)

**개선사항:**
- [ ] (fidelity) MSP 기반 selection 정의(원논문) vs 구현(현재 MSP drop 기반) 일치성 확인 및 필요시 수정

---

### 3. CloudLeak (`mebench/attackers/cloudleak.py`)

**문제점:**
- (stale) 이전 TODO에는 FeatureFool objective gradient 미반환 이슈가 있었으나, 현재 구현은 `(loss, grad)`를 반환함(재검증/정리 필요)

**개선사항:**
- [ ] L-BFGS 호출부에서 `factr/pgtol`이 config 값을 사용하도록 정리
- [ ] `epsilon`을 config로 노출

---

### 4. MAZE (`mebench/attackers/maze.py`)

**문제점:**
- `propose(k)`가 `k*(m+1)` 이미지를 반환(계약 위반 가능성)

---

### 5. Evaluator (`mebench/eval/evaluator.py`)

**문제점:**
- `num_classes=10`이 하드코딩됨 (Line 92)
- Dataset config에서 `num_classes`를 읽어오도록 개선 필요

**개선사항:**
- [ ] Hardcoded `num_classes=10`를 dataset config에서 읽어오도록 변경
  - `state.metadata.get("num_classes", 10)` 또는 dataset config에서 읽음
  - 이것은 CIFAR-100(100 classes) 등 다른 데이터셋에서 필요함

---

### 6. Data Loader (`mebench/data/loaders.py`)

**문제점:**
- Data-free mode에서 `NotImplementedError("Data-free mode uses synthetic queries")`를 발생시킴 (Line 142)
- 현재 DFME, MAZE, CloudLeak 같은 data-free 공격들이지만 loader가 이를 처리하지 못함

**개선사항:**
- [ ] Data-free mode 지원
  - Data-free mode에서는 실제 데이터셋을 로드하지 않고, 공격자가 합성 쿼리를 생성함
  - Loader가 data-free mode에서는 빈 dataset 또는 synthetic data를 생성할 수 있도록 수정 필요

---

## 구조적 개선사항 (Project Level)

### Algorithm Fidelity

**SwiftThief:**
- [ ] 전체 SimSiam 알고리즘 구현 (positive/negative pairs)
- [ ] Entropy-weighted 대조 손실 정확한 수식 구현
- [ ] Real data augmentation (torchvision.transforms)

**Black-box Dissector:**
- [ ] Layer-wise Grad-CAM (last conv feature hooks)
- [ ] Connected component-based erasing

**CloudLeak:**
- [ ] PyTorch autograd 기반 FeatureFool (gradient-based L-BFGS)
- [ ] 또는 finite-difference gradient (faithful to paper)

### Code Quality

- [ ] 모든 `pass` placeholder 제거
- [ ] 모든 `NotImplementedError` 제거 또는 구현
- [ ] Unused import 제거
- [ ] Type hints 강화 (return types 완전화)
- [ ] `num_classes` 하드코딩 제거 (attackers/evaluator/victim_loader/substitute_factory)

### Testing

- [ ] End-to-end integration test for each new attack
- [ ] Budget accounting test (Dissector 2x budget 정확히 계산되는지)
- [ ] Reproducibility test (fixed seed → identical results)

---

## Environment Setup (Remaining)

1. **로컬 테스트/진단 실행 환경 확인**
   - `pytest` 설치/경로 확인 후 신규 테스트 실행
   - `basedpyright` 설치 후 LSP diagnostics 실행

---

## Paper Fidelity Review (Jan 22, 2026)

### SwiftThief (`mebench/attackers/swiftthief.py` vs `swiftthief.pdf`)
- [x] Cost-sensitive sharpness (class imbalance weighting) 적용
- [x] Rare-class sampling 전환 조건 구현 (budget-based switch)
- [x] class-balanced weighting (inverse frequency)

### Black-box Dissector (`mebench/attackers/blackbox_dissector.py` vs `blackbox-dissector.pdf`)
- [x] N개 erasing 중 confidence 최대 감소 variant 선택 로직 추가
- [x] Erasing ratio 기본값 0.25 적용
- [x] Erasing ratio schedule 지원 (linear, optional)
- [x] PrioriPatchErasing 확률 분포 기반 center sampling 적용

### CloudLeak (`mebench/attackers/cloudleak.py` vs `cloudleak.pdf`)
- [x] Margin M (Eq.11) 계산 및 class intra-distance 반영
- [x] Layer-wise feature 선택(ResNet avgpool / VGG-AlexNet classifier[4]) 적용
- [x] FeatureFool target layer configurable via attack.feature_layer

### MAZE (`mebench/attackers/maze.py` vs `MAZE.pdf`)
- [x] Budget accounting 기반 scheduler t_max 계산 적용
- [x] KL divergence uses log_softmax(logits) vs victim probs
- [x] Replay buffer: uniform sampling + max size cap
- [x] CosineAnnealingLR 스케줄러 적용 (budget 기반)

### DFME (`mebench/attackers/dfme.py` vs `DFME.pdf`)
- [x] Generator gradient가 pre-tanh input 기준으로 적용
- [x] LR schedule(0.3 decay at 10/30/50%) 구현 (query budget 기준)
- [ ] Student/Generator update ratio와 예산 회계 일치성 검증

### DFMS-HL (`mebench/attackers/dfms.py` vs `DFMS.pdf`)
- [ ] DCGAN 구조가 논문과 동일한지 확인 (5-layer TransposedConv)
- [x] Diversity coefficient (lambda_div 500/100) 적용
- [x] Clone pretrain 200 epochs 및 cosine LR 스케줄 적용

### GAME (`mebench/attackers/game.py` vs `GAME.pdf`)
- [x] AC-GAN 기반 TDL 사전학습 (discriminator aux classifier 활용)
- [x] ACS 계산 (P_unc/P_dev) normalized via min-shift + sum
- [ ] Offline retrain 단계 추가 여부

### ESAttack (`mebench/attackers/es_attack.py` vs `esattack.pdf`)
- [x] OPT-SYN: Dirichlet target 반영
- [x] OPT-SYN: augmentation(Flip/Affine/Gaussian) 적용 (일반적 설정)
- [x] DNN-SYN: class-conditional generator + ACGAN-style CE + mode-seeking loss
- [x] E-step epoch schedule 기본값 10 적용

### InverseNet (`mebench/attackers/inversenet.py` vs `inversenet.pdf`)
- [x] Coreset selection (phase 1) 구현 (L1 K-center greedy)
- [x] HCSS(High Confidence Score Sampling) 구현 (DeepFool perturbation 기반)
- [x] Truncated logits + inversion augmentation pipeline 구현

### KnockoffNets (`mebench/attackers/knockoff_nets.py` vs `knockoffnets.pdf`)
- [x] Reward: certainty + diversity + loss 복합 구성
- [ ] Offline retrain 단계 및 pretrained init 반영
- [ ] Hierarchical bandit policy 구조 검증

### Missing attacks
- [x] Blackbox Ripper (`blackbox-ripper.pdf`) 구현 추가
- [x] CopycatCNN (`copycatcnn.pdf`) 구현 추가

---

## Assumption Log (Needs mention in paper/code)
- SwiftThief: cost-sensitive sharpness implemented via class-frequency weighted CE + FGSM regularizer.
- Black-box Dissector: PrioriPatchErasing center sampled from Grad-CAM heatmap distribution; erasing ratio default 0.25; variant score uses MSP drop.
- CloudLeak: layer mapping set to ResNet avgpool / VGG-AlexNet classifier[4] / last linear fallback.
- MAZE: budget accounting used to set scheduler horizon (t_max) from query-per-iter approximation.
- ESAttack: mode-seeking loss uses MSGAN-style ratio (||G(z1)-G(z2)|| / ||z1-z2||); ACGAN not implemented.
- ESAttack: OPT-SYN augmentation uses generic flip/affine/gaussian noise (parameters not specified in paper).
- GAME: discriminator uses auxiliary classifier (ACGAN) with DCGANDiscriminator class head.
- DFME: generator exposes pre-tanh output; queries still use tanh output (pre-tanh gradient handling not explicitly separated).
- DFME: pre-tanh gradient now applied to generator update; perturbations generated in pre-tanh space then tanh.
- DFMS-HL: DCGAN generator depth remains 3 upsampling blocks for 32x32 outputs (paper not explicit for this resolution).
- DCGAN: generator/discriminator now scale with input/output size using num_upsamples from output_size; default remains 32x32.
- ESAttack: ACGAN-style CE applied via student logits (no explicit discriminator).

---

## Hyperparameter Defaults Summary

### SwiftThief
```yaml
attack:
  name: "swiftthief"
  I: 10
  initial_seed_ratio: 0.1
  lambda1: 1.0
  lambda2: 0.01
  lambda3: 1.0
  fgsm_epsilon: 0.01
  projection_dim: 2048
  batch_size: 256
  lr: 0.001
  momentum: 0.9
  weight_decay: 5e-4
```

### Black-box Dissector
```yaml
attack:
  name: "blackbox_dissector"
  n_variants: 10
  erasing_ratio: 0.25
  batch_size: 128
  max_epochs: 1000
  patience: 100
  lr: 0.001
  dropout: 0.1
  l2_reg: 0.001
```

### CloudLeak
```yaml
attack:
  name: "cloudleak"
  lbfgs_iters: 20
  lbfgs_factr: 10000000.0
  lbfgs_pgtol: 1e-05
  margin_m: 0.5
  lambda_adv: 0.001
  initial_pool_size: 1000
  batch_size: 64
  lr: 0.01
  momentum: 0.9
  weight_decay: 5e-4
```

---

## Attack Fidelity Audit (Jan 23, 2026)

This section logs **paper/repo mismatches** found while auditing `mebench/attackers/` against the corresponding PDFs in `papers/` and (when available) the official implementations.

Conventions:
- **[CRITICAL]**: algorithm/contract violation or obvious training stub
- **[MAJOR]**: materially different defaults/architecture/training schedule vs paper/repo
- **[MINOR]**: small deviation/parameterization/robustness issue

### ActiveThief (`mebench/attackers/activethief.py` vs `papers/activethief.pdf`)
- [ ] **[MAJOR]** `DFAL+K-center` hybrid strategy is documented but not implemented.
  - Code: `mebench/attackers/activethief.py:33` (doc mentions), `mebench/attackers/activethief.py:46` (strategy options omit hybrid)
  - Paper: `papers/activethief.pdf` (subset selection strategies; DFAL+K-center described)
  - Official repo: `https://github.com/iisc-seal/activethief` (implements combo strategy)
- [ ] **[MAJOR]** Retrain cadence is hardcoded to 100-query intervals.
  - Code: `mebench/attackers/activethief.py:398` (`if labeled_count % 100 == 0`)
  - Paper: iteration/round-based loop; step size should be configurable.

### SwiftThief (`mebench/attackers/swiftthief.py` vs `papers/swiftthief.pdf`)
- [ ] **[MAJOR]** Retrain cadence is hardcoded to 100-query intervals (should be driven by paper’s iteration schedule).
  - Code: `mebench/attackers/swiftthief.py:549-551`
  - Paper: `papers/swiftthief.pdf` (outer iterations `I` and per-iteration epochs)
- [ ] **[MINOR]** Contrastive/augmentation pipeline fidelity should be re-checked against any official repo; earlier TODO entries about placeholders appear stale.
  - Code check: no `pass` placeholders in `mebench/attackers/swiftthief.py` (grep)

### Black-box Dissector (`mebench/attackers/blackbox_dissector.py` vs `papers/blackbox-dissector.pdf`)
- [ ] **[MAJOR]** Selection metric ranks samples by **MSP drop** (original MSP − erased MSP), but the paper’s query strategy describes selecting erased samples by **high MSP**.
  - Code: `mebench/attackers/blackbox_dissector.py:323-362` (computes `drop` and sorts by it)
  - Paper: `papers/blackbox-dissector.pdf` (Query strategy; MSP-based selection of erased images)
- [ ] **[MAJOR]** Pool-exhaustion fallback returns random noise with hardcoded CIFAR shape and doubles the requested k.
  - Code: `mebench/attackers/blackbox_dissector.py:302-305` (`torch.randn(2 * k, 3, 32, 32)`)

### Black-box Ripper (`mebench/attackers/blackbox_ripper.py` vs `papers/blackbox-ripper.pdf`)
- [ ] **[CRITICAL]** Substitute training is a one-batch stub due to an unconditional `break`.
  - Code: `mebench/attackers/blackbox_ripper.py:246-263` (break at `:260`)
- [ ] **[MAJOR]** Latent initialization differs from paper (Normal vs Uniform U(-u, u)); defaults (population/elite) differ.
  - Code: `mebench/attackers/blackbox_ripper.py:108-110` (population `torch.randn`, random targets)
  - Paper: `papers/blackbox-ripper.pdf` Algorithm 1 (population sampled from `U(-u,u)`, e.g., u=3)
- [ ] **[MAJOR]** Generator model family differs (DCGAN here vs ProGAN/SNGAN in paper).
  - Code: `mebench/attackers/blackbox_ripper.py:60-66`
  - Paper: `papers/blackbox-ripper.pdf` (experimental setup: ProGAN/SNGAN)

### CloudLeak (`mebench/attackers/cloudleak.py` vs `papers/cloudleak.pdf`)
- [ ] **[MAJOR]** The docstring claims “select samples with highest uncertainty”, but the propose path should be audited to confirm there is an explicit uncertainty-based filtering stage.
  - Code: `mebench/attackers/cloudleak.py:197-210` (doc)
- [ ] **[MAJOR]** Transfer learning is enforced by freezing the backbone.
  - Code: `mebench/attackers/cloudleak.py:515-519`
- [ ] **[MAJOR]** L-BFGS parameters in the call are hardcoded instead of using configured values.
  - Code: `mebench/attackers/cloudleak.py:176-183` (uses constants), config fields exist at `:232-235`
- [ ] **[MAJOR]** `epsilon` is hardcoded to `8/255` and not configurable.
  - Code: `mebench/attackers/cloudleak.py:240`
  - Official repo: `https://github.com/yunyuntsai/DNN-Model-Stealing`

### CopycatCNN (`mebench/attackers/copycatcnn.py` vs `papers/copycatcnn.pdf`)
- [ ] **[CRITICAL]** Substitute training is a one-batch stub due to an unconditional `break`.
  - Code: `mebench/attackers/copycatcnn.py:136-145` (break at `:144`)
- [ ] **[MAJOR]** Pool exhaustion falls back to Gaussian noise; the paper uses natural NPD images.
  - Code: `mebench/attackers/copycatcnn.py:63-69`
  - Paper: `papers/copycatcnn.pdf` (transfer set from NPD datasets)
- [ ] **[MAJOR]** Offline augmentation pipeline from the paper/official codebase is not implemented.
  - Official repo: `https://github.com/jeiks/Stealing_DL_Models`

### KnockoffNets (`mebench/attackers/knockoff_nets.py` vs `papers/knockoffnets.pdf`)
- [ ] **[MAJOR]** Reward definitions and bandit update differ from the paper’s gradient-bandit equations.
  - Paper: `papers/knockoffnets.pdf` (gradient bandit update; certainty/diversity/loss reward definitions)
- [ ] **[MAJOR]** Pool exhaustion falls back to random noise with hardcoded CIFAR shape.
  - Code: `mebench/attackers/knockoff_nets.py:208-210`
- [ ] **[MAJOR]** Implementation assumes pool dataset has labels to define “actions”; paper allows unlabeled $P_A$ with victim posteriors.
  - Official repo: `https://github.com/tribhuvanesh/knockoffnets`

### DFME (`mebench/attackers/dfme.py` vs `papers/DFME.pdf`)
- [ ] **[MAJOR]** Generator optimizer LR default differs from official repo defaults (paper commonly cites 5e-4; official repo uses 1e-4).
  - Code: `mebench/attackers/dfme.py:46`
  - Official repo: `https://github.com/cake-lab/datafree-model-extraction`
- [ ] **[MAJOR]** Generator architecture is generic DCGAN; paper describes a specific 3-layer + linear upsampling design.
  - Code: `mebench/attackers/dfme.py:98-106` (`DCGANGenerator`)

### DFMS-HL (`mebench/attackers/dfms.py` vs `papers/DFMS.pdf`)
- [ ] **[MAJOR]** Clone LR default is `0.01` here, while paper settings commonly use `0.1` (SGD) for the clone.
  - Code: `mebench/attackers/dfms.py:23`
- [ ] **[MAJOR]** Requires `proxy_dataset` (proxy-data variant); verify against intended DFMS-HL setting and official repo.
  - Code: `mebench/attackers/dfms.py:98-104` (proxy required)
  - Official repo: `https://github.com/val-iisc/Hard-Label-Model-Stealing`

### GAME (`mebench/attackers/game.py` vs `papers/GAME.pdf`)
- [ ] **[MAJOR]** Generator/discriminator architecture should be checked against paper (GAME describes specific AC-GAN style blocks; current implementation uses shared DCGAN modules).
  - Paper: `papers/GAME.pdf` (generator/discriminator architecture)
- [ ] **[MAJOR]** Generator loss weights (`beta_i`) are sensitive and are currently defaulted; verify they match paper/official repo defaults.
  - Paper: `papers/GAME.pdf` (Loss_Total weights)
  - Official repo: `https://github.com/xythink/game-attack`

### MAZE (`mebench/attackers/maze.py` vs `papers/MAZE.pdf`)
- [ ] **[CRITICAL]** `propose(k)` returns `k * (m + 1)` images (base + perturbed) rather than exactly k.
  - Code: `mebench/attackers/maze.py:94-113`
- [ ] **[MAJOR]** Default learning rates differ from paper/official repo (e.g., clone LR here 0.01 vs paper 0.1).
  - Code: `mebench/attackers/maze.py:23-30`
  - Official repo: `https://github.com/sanjaykariyappa/MAZE`

### ESAttack (`mebench/attackers/es_attack.py` vs `papers/esattack.pdf`)
- [ ] **[MAJOR]** `OPT-SYN` optimization runs in `observe()` after querying.
  - Code: `mebench/attackers/es_attack.py:86-109`, `mebench/attackers/es_attack.py:130-134`
  - Paper: `papers/esattack.pdf` Algorithm 1 (E-step / S-step)
- [ ] **[MAJOR]** Synthetic buffer uses hardcoded CIFAR shape `(3, 32, 32)` instead of `state.metadata["input_shape"]`.
  - Code: `mebench/attackers/es_attack.py:81-85`

### InverseNet (`mebench/attackers/inversenet.py` vs `papers/inversenet.pdf`)
- [ ] **[CRITICAL]** Inversion training has a one-batch stub due to an unconditional `break`.
  - Code: `mebench/attackers/inversenet.py:174-185` (break at `:184`)
- [ ] **[MAJOR]** Query-dataset substitute training has a one-batch stub due to an unconditional `break`.
  - Code: `mebench/attackers/inversenet.py:280-291` (break at `:290`)
- [ ] **[MAJOR]** Truncation default is top-3 (`truncation_k=3`), but paper’s restrictive setting uses top-1.
  - Code: `mebench/attackers/inversenet.py:35`

### RandomBaseline (`mebench/attackers/random_baseline.py`)
- [ ] **[MAJOR]** Loads `dataset_config` from `state.attack_state` but engine conventions use `state.metadata["dataset_config"]`.
  - Code: `mebench/attackers/random_baseline.py:113`
- [ ] **[MAJOR]** Pool-exhaustion fallback returns random noise with hardcoded CIFAR shape.
  - Code: `mebench/attackers/random_baseline.py:118-122`
