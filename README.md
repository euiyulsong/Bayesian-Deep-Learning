# Bayesian / Probabilistic Deep Learning Benchmark

## 1. 실험 개요

본 실험은 동일한 1D regression task에서 다음 probabilistic / Bayesian 계열 모델을 비교한다.

* Deterministic MLP
* MC Dropout
* Deep Ensemble
* Mean-field Bayesian Neural Network
* Exact Gaussian Process
* Conditional Neural Process
* Latent Neural Process

데이터는 RBF kernel 기반 Gaussian Process에서 생성된 함수이며, 일부 context point만 관측하고 전체 구간의 함수를 예측한다.

개념적으로는 다음 문제다.

```text
                 observed context
              ●          ●
       ●                         ●
------------------------------------------------ x

        ???????????????????????
          unknown function
```

모델은 관측된 소수의 점으로부터

$$
p(y_* \mid x_*, D)
$$

즉 새로운 위치 \(x_*\)에서의 **prediction뿐 아니라 uncertainty까지** 예측해야 한다.

---

# 2. 최종 결과

| Model             |    RMSE ↓ |        NLL ↓ | Coverage@95% | Interval Width ↓ | Latency ms ↓ |
| ----------------- | --------: | -----------: | -----------: | ---------------: | -----------: |
| **Exact GP**      |     0.622 |    **0.147** |    **1.000** |            1.621 |    **0.049** |
| Neural Process    |     0.875 |        1.357 |        0.824 |            2.742 |       27.014 |
| CNP               |     0.985 |        1.667 |        0.781 |            2.556 |        0.465 |
| BNN Mean-field VI |     0.847 |        2.784 |        0.719 |            1.342 |       19.772 |
| MC Dropout        |     0.440 |        6.010 |        0.746 |            0.740 |        7.983 |
| Deep Ensemble     |     0.386 |        9.270 |        0.688 |            0.758 |        0.870 |
| **MLP**           | **0.275** | **1353.499** |        0.609 |        **0.487** |        0.248 |

결과가 매우 재미있다.

```text
Point Prediction Ranking
────────────────────────────────────
MLP
  ↓
Deep Ensemble
  ↓
MC Dropout
  ↓
GP
  ↓
BNN
  ↓
NP
  ↓
CNP


Uncertainty / NLL Ranking
────────────────────────────────────
Exact GP
  ↓
NP
  ↓
CNP
  ↓
BNN
  ↓
MC Dropout
  ↓
Ensemble
  ↓
MLP
```

즉 거의 정반대다.

---

# 3. 가장 중요한 결론

## RMSE가 좋다고 uncertainty가 좋은 것이 아니다

MLP의 RMSE는

$$
0.275
$$

로 모든 모델 중 가장 좋다.

하지만 NLL은

$$
1353.5
$$

로 사실상 망가졌다.

왜냐하면 MLP가 prediction 자체는 맞게 하지만

$$
\sigma(x)
$$

를 지나치게 작게 예측하기 때문이다.

Gaussian predictive likelihood는

$$
p(y\mid x)
=
\mathcal N
\left(
y;
\mu(x),
\sigma^2(x)
\right)
$$

이고 NLL은

$$
-\log p(y\mid x)
=
\frac12
\log(2\pi\sigma^2)
+
\frac{(y-\mu)^2}{2\sigma^2}.
$$

여기서 \(\sigma\)가 지나치게 작으면 작은 prediction error조차

$$
\frac{(y-\mu)^2}{\sigma^2}
$$

에 의해 크게 증폭된다.

예를 들어

```text
Truth     : 1.0
Prediction: 0.9

Case A: std = 0.5
████████████████████

Case B: std = 0.01
          █
```

prediction error는 똑같이 0.1이지만 Case B는

> "나는 0.9가 거의 확실하다."

라고 주장했기 때문에 훨씬 큰 penalty를 받는다.

MLP의 결과가 정확히 이것이다.

---

# 4. Deterministic MLP

## 구조

```text
x
│
▼
Linear
│
Tanh
│
▼
Linear
│
Tanh
│
▼
hidden representation
 ├──────────────┐
 ▼              ▼
mean head     std head
 μ(x)           σ(x)
```

예측 distribution:

$$
p(y|x,\theta)
=
\mathcal N
\left(
\mu_\theta(x),
\sigma_\theta^2(x)
\right)
$$

loss:

$$
\mathcal L
=
-\sum_i
\log
p(y_i|x_i,\theta)
$$

이다.

하지만 parameter \(\theta\) 자체는 point estimate다.

$$
p(\theta|D)
\approx
\delta(\theta-\hat\theta).
$$

따라서

> parameter uncertainty = 없음.

### 결과

$$
RMSE=0.275
$$

로 가장 좋았지만

$$
Coverage=60.9\%
$$

밖에 되지 않는다.

95% confidence interval을 출력했는데 실제 point의 61%만 포함한다.

즉 극심한 **overconfidence**다.

```text
prediction
───────────────

95% interval
       ├───┤

truth
              ●
```

Interval width도

$$
0.487
$$

로 전체 모델 중 가장 좁다.

따라서

> MLP는 mean function fitting에는 매우 강했지만 epistemic uncertainty를 표현하지 못했다.

---

# 5. MC Dropout

MC Dropout은 dropout mask를 inference에서도 유지한다.

각 forward pass마다

$$
\theta_t
$$

가 조금씩 다른 network처럼 행동한다.

```text
       dropout mask #1 → μ₁
x ──── dropout mask #2 → μ₂
       dropout mask #3 → μ₃
       ...
       dropout mask #T → μT
```

predictive mean:

$$
\hat\mu
=
\frac1T
\sum_{t=1}^T
\mu_t.
$$

epistemic variance:

$$
\sigma_{\mathrm{epi}}^2
=
\frac1T
\sum_t
(\mu_t-\hat\mu)^2.
$$

aleatoric variance까지 포함하면

$$
\sigma_{\mathrm{total}}^2
=
\underbrace{
\frac1T\sum_t\sigma_t^2
}_{\text{aleatoric}}
+
\underbrace{
\operatorname{Var}_t(\mu_t)
}_{\text{epistemic}}.
$$

### 결과

$$
RMSE=0.440
$$

로 상당히 좋다.

하지만

$$
Coverage=74.6\%
$$

로 여전히 95%에 크게 못 미친다.

Interval width:

$$
0.740
$$

역시 좁다.

즉

> Dropout이 uncertainty를 추가하기는 하지만 이 실험에서는 epistemic uncertainty를 충분히 크게 만들지 못했다.

---

# 6. Deep Ensemble

Deep Ensemble은 서로 다른 initialization과 bootstrap dataset으로 여러 모델을 학습한다.

```text
             MLP #1 ─── μ₁, σ₁
           /
Data ───── MLP #2 ─── μ₂, σ₂
           \
             MLP #3 ─── μ₃, σ₃
```

predictive mean:

$$
\mu
=
\frac1M
\sum_{m=1}^{M}
\mu_m.
$$

전체 variance:

$$
\sigma^2
=
\underbrace{
\frac1M
\sum_m \sigma_m^2
}_{aleatoric}
+
\underbrace{
\frac1M
\sum_m
(\mu_m-\mu)^2
}_{epistemic}.
$$

### 결과

$$
RMSE=0.386
$$

으로 MLP 다음으로 좋다.

하지만

$$
Coverage=68.8\%
$$

이다.

즉 ensemble member 세 개가 서로 상당히 비슷한 solution에 수렴했다는 의미다.

```text
Model 1 ───────────
Model 2 ───────────
Model 3 ───────────
          거의 동일
```

Quick 설정이라

$$
M=3
$$

뿐인 것도 영향을 줬을 가능성이 크다.

Deep Ensemble은 모델 수를

$$
3 \rightarrow 5 \rightarrow 10
$$

으로 늘려보면 calibration이 개선되는지 확인할 가치가 있다.

---

# 7. Bayesian Neural Network

BNN은 weight 자체를 distribution으로 둔다.

Deterministic NN:

$$
w = \hat w
$$

BNN:

$$
w \sim q_\phi(w).
$$

본 실험에서는 mean-field Gaussian posterior를 사용한다.

$$
q_\phi(w)
=
\prod_i
\mathcal N
(w_i;\mu_i,\sigma_i^2).
$$

Prior는

$$
p(w)=\mathcal N(0,1).
$$

Variational inference objective는 ELBO:

$$
\mathcal L_{\mathrm{ELBO}}
=
\mathbb E_{q(w)}
[
\log p(D|w)
]
-
KL
\left(
q(w)
\Vert
p(w)
\right).
$$

loss 관점에서는

$$
\mathcal L
=
-\mathbb E_{q(w)}
[\log p(D|w)]
+
\beta KL(q(w)||p(w)).
$$

구조:

```text
             weight distribution

               N(μw, σw)
                    │
                    ▼
x ─────────── Bayesian Layer
                    │
                 sample w
                    │
                    ▼
               prediction
```

### 결과

$$
RMSE=0.847
$$

로 prediction 성능은 떨어졌지만

$$
NLL=2.784
$$

로 MLP/Ensemble/Dropout보다 훨씬 낫다.

Coverage:

$$
71.9\%
$$

이다.

다만 아직 이상적인 95%와 거리가 있다.

가장 큰 이유는 **mean-field approximation의 한계**다.

실제 posterior:

```text
w1
▲
│       ╱████
│    ╱████
│ ╱████
└────────────► w2

correlated posterior
```

Mean-field:

```text
w1
▲
│     ███
│     ███
│     ███
└────────────► w2

q(w1,w2)=q(w1)q(w2)
```

즉 parameter correlation을 표현할 수 없다.

---

# 8. Gaussian Process

이번 실험의 승자라고 볼 수 있다.

GP는 neural network처럼 parameter를 직접 Bayesian하게 만드는 게 아니라

> **함수 자체에 distribution을 둔다.**

$$
f(x)
\sim
GP(m(x),k(x,x')).
$$

RBF kernel:

$$
k(x,x')
=
\sigma_f^2
\exp
\left(
-\frac{(x-x')^2}{2\ell^2}
\right).
$$

즉 가까운 \(x\)끼리는 함수값이 비슷할 것이라는 prior다.

```text
Function samples from GP prior

   ~~~~~~~~
 ~~        ~~~
       ~~~~~~~~
~~~~
------------------------ x
```

관측값

$$
X,\quad y
$$

가 있을 때 test point \(X_*\)에 대한 posterior는 analytic하게 계산된다.

$$
\mu_*
=
K_{*X}
(K_{XX}+\sigma_n^2I)^{-1}y
$$

그리고

$$
\Sigma_*
=
K_{**}
-
K_{*X}
(K_{XX}+\sigma_n^2I)^{-1}
K_{X*}.
$$

직관적으로:

```text
Few observations

     ●
            ●

posterior mean
────────╲____╱────────

uncertainty
██████▓▓░░   ░░▓▓██████
```

관측점 근처:

$$
\sigma(x)\downarrow
$$

관측점에서 멀어지면:

$$
\sigma(x)\uparrow.
$$

### 결과

$$
NLL=0.147
$$

로 압도적인 1위.

Coverage:

$$
100\%.
$$

Interval width:

$$
1.621.
$$

즉 GP가 다른 모델보다 uncertainty를 훨씬 보수적으로 추정했다.

다만 target이 95%인데 coverage가 100%이므로 약간 **under-confident**, 즉 interval이 조금 넓다고 볼 수도 있다.

가장 흥미로운 점은

$$
RMSE=0.622
$$

라는 것이다.

MLP보다 mean prediction은 훨씬 나쁜데 probabilistic prediction은 훨씬 좋다.

즉

> RMSE와 probabilistic quality는 다른 objective다.

---

# 9. Conditional Neural Process

CNP는 GP처럼

$$
\text{context set}
\rightarrow
\text{function prediction}
$$

을 수행하지만 이를 neural network로 amortize한다.

각 context point:

$$
(x_i,y_i)
$$

를 encoder에 넣는다.

$$
r_i
=
h_\theta(x_i,y_i)
$$

그리고 aggregation:

$$
r
=
\frac1N
\sum_i r_i.
$$

target point에서는

$$
p(y_*|x_*,r)
=
\mathcal N
(\mu(x_*,r),\sigma^2(x_*,r)).
$$

구조:

```text
(x1,y1) ── Encoder ─┐
(x2,y2) ── Encoder ─┼── Mean ─── r
(x3,y3) ── Encoder ─┘
                           │
                  x* ─────┤
                           ▼
                       Decoder
                           │
                       μ*, σ*
```

이 구조의 핵심 장점은 permutation invariance다.

$$
\{(x_1,y_1),(x_2,y_2)\}
$$

와 순서를 뒤집은 context가 동일한 representation을 만든다.

### 결과

$$
RMSE=0.985
$$

$$
NLL=1.667
$$

$$
Coverage=78.1\%.
$$

RMSE는 좋지 않지만 calibration은 deterministic 계열보다 상당히 좋다.

중요한 이유는 CNP가 단일 함수에 fitting된 게 아니라

> 여러 GP function을 반복해서 보고
> “context가 이런 모양이면 나머지 함수가 어떤 식일 가능성이 높은가”

를 학습했다는 점이다.

즉 **amortized function inference**다.

---

# 10. Latent Neural Process

NP는 CNP에 latent variable을 추가한다.

CNP:

$$
r = \operatorname{Aggregate}(h(x_i,y_i)).
$$

NP에서는 추가로

$$
z \sim q(z|C)
$$

를 사용한다.

```text
Context
  │
  ├──── deterministic encoder ─── r
  │
  └──── latent encoder ───────── q(z|C)
                                  │
                                  ▼
                              sample z
                                  │
                  x* ─── r ─── z
                         │
                         ▼
                      Decoder
                         │
                      μ*, σ*
```

학습 시에는 target까지 본 posterior

$$
q(z|T)
$$

와 context-only prior

$$
q(z|C)
$$

를 맞춘다.

ELBO 형태는

$$
\mathcal L
=
\mathbb E_{q(z|T)}
[
\log p(y_T|x_T,z,C)
]
-
KL
\left(
q(z|T)
\Vert
q(z|C)
\right).
$$

실험 로그에서도 이를 볼 수 있다.

초기:

$$
KL=2.289
$$

500 step:

$$
KL=0.539.
$$

즉 training이 진행되면서

$$
q(z|T)
\approx q(z|C)
$$

가 되어 가고 있다.

### 결과

$$
RMSE=0.875
$$

$$
NLL=1.357
$$

$$
Coverage=82.4\%.
$$

CNP보다

* RMSE 개선
* NLL 개선
* Coverage 개선

이 모두 나타났다.

즉 이번 실험에서는 latent variable이 실제로 도움이 됐다.

```text
                   CNP       NP
RMSE              0.985 → 0.875
NLL               1.667 → 1.357
Coverage           78.1 → 82.4%
```

NP가 CNP보다 uncertainty를 더 잘 표현한 이유는

$$
z
$$

가 **global function uncertainty**를 표현하기 때문이다.

---

# 11. GP와 Neural Process의 관계

이 둘이 사실 이번 실험에서 가장 재미있는 비교다.

## Gaussian Process

매 새로운 task마다 Bayesian inference:

$$
D
\rightarrow
p(f|D).
$$

## Neural Process

많은 task를 미리 학습한 뒤

$$
D
\rightarrow
q_\theta(f|D)
$$

를 neural network 한 번으로 근사한다.

```text
                    GP
new context ─────────────► exact Bayesian inference
                           matrix operations
                           │
                           ▼
                         posterior


                    NP
many training functions
        │
        ▼
   learn inference network
        │
        ▼

new context ─────► neural network ─────► approximate posterior
```

즉 NP는

> **amortized Gaussian Process-like inference**

라고 볼 수 있다.

이번 latency:

$$
GP=0.049ms
$$

$$
CNP=0.465ms
$$

이지만 이는 context가 겨우 12개인 tiny setting이라 GP가 매우 유리한 것이다.

GP complexity는 일반적으로 dense exact inference에서 training/factorization 기준

$$
O(N^3)
$$

이고 memory:

$$
O(N^2).
$$

반면 CNP/NP는 학습 이후 context 처리 비용이 훨씬 유리하게 확장될 수 있다.

---

# 12. 왜 GP의 RMSE가 MLP보다 나쁜가?

처음 보면 이상해 보인다.

```text
MLP RMSE = 0.275
GP  RMSE = 0.622
```

하지만 현재 GP kernel hyperparameter가 실제 generator와 맞춰져 있어도 context point가 12개뿐이다.

GP는 관측하지 않은 영역에서 prior로 돌아간다.

예:

```text
observed                         observed
   ●                                ●
   │                                │

────╲                              ╱────
     ╲____________________________╱
                GP prior
```

반면 MLP는 관측된 점을 이용해 강하게 extrapolation할 수 있다.

우연히 target function의 shape와 extrapolation이 잘 맞으면 RMSE가 매우 낮아질 수 있다.

하지만 MLP는

> 내가 안 본 영역이라는 사실

을 거의 인식하지 못한다.

그래서

```text
Accuracy      매우 좋음
Confidence    지나치게 높음
```

이라는 결과가 나온다.

---

# 13. Coverage와 Width를 같이 봐야 하는 이유

95% interval coverage 목표는

$$
P
\left(
y\in[\mu-1.96\sigma,\mu+1.96\sigma]
\right)
\approx0.95.
$$

하지만 coverage만 높으면 항상 좋은 것도 아니다.

예를 들어

```text
Model A

|----------------------------------------------|
                    ●

coverage = 100%
```

interval을 무한히 넓히면 coverage는 항상 100%가 된다.

따라서 함께 봐야 한다.

$$
Coverage \rightarrow 0.95
$$

하면서

$$
IntervalWidth \downarrow.
$$

이번 결과를 보면:

| Model      | Coverage | Width |
| ---------- | -------: | ----: |
| GP         |    1.000 | 1.621 |
| NP         |    0.824 | 2.742 |
| CNP        |    0.781 | 2.556 |
| BNN        |    0.719 | 1.342 |
| MC Dropout |    0.746 | 0.740 |
| Ensemble   |    0.688 | 0.758 |
| MLP        |    0.609 | 0.487 |

NP는 interval이 **GP보다 훨씬 넓은데도 coverage는 더 낮다.**

이건 단순히 uncertainty의 크기만 문제가 아니라

> uncertainty가 필요한 위치에 제대로 배치되지 않았다는 것

을 뜻한다.

즉 calibration의 spatial structure가 아직 좋지 않다.

---

# 14. NLL이 보여주는 것

RMSE:

$$
\sqrt{
\frac1N
\sum_i
(y_i-\mu_i)^2
}
$$

는 mean prediction만 본다.

반면 NLL:

$$
-\frac1N
\sum_i
\log
p(y_i|x_i)
$$

은

$$
\mu_i
$$

와

$$
\sigma_i
$$

를 동시에 평가한다.

따라서 probabilistic model에서는 NLL이 훨씬 중요하다.

이번 결과:

```text
                         RMSE       NLL
MLP                      ★★★★★      ☆
Deep Ensemble            ★★★★       ☆
MC Dropout               ★★★        ★
Exact GP                 ★★         ★★★★★
Neural Process           ★          ★★★★
```

정도로 볼 수 있다.

---

# 15. Bayesian 관점에서 모델 분류

```text
Probabilistic Deep Learning
│
├── Deterministic Parameters
│   │
│   ├── Gaussian MLP
│   │
│   └── CNP
│
├── Approximate Parameter Uncertainty
│   │
│   ├── MC Dropout
│   ├── Deep Ensemble
│   └── BNN Variational Inference
│
├── Function-space Bayesian Model
│   │
│   └── Gaussian Process
│
└── Latent Function Distribution
    │
    └── Neural Process
```

조금 더 엄밀하게 보면 Deep Ensemble은 Bayesian inference는 아니지만 Bayesian model averaging과 비슷한 효과를 노리는 방법이다.

---

# 16. 전체 결과의 핵심 해석

이번 실험에서는 세 그룹이 분명하게 갈렸다.

## Group A — Prediction specialist

```text
MLP
Deep Ensemble
MC Dropout
```

RMSE는 좋다.

하지만 uncertainty가 과도하게 작다.

$$
\text{high accuracy}
+
\text{poor calibration}
$$

이다.

---

## Group B — Approximate Bayesian models

```text
BNN
CNP
Neural Process
```

mean prediction은 떨어지지만 uncertainty quality는 개선된다.

특히 NP가

$$
NLL=1.357
$$

로 approximation 기반 모델 중 가장 좋았다.

---

## Group C — Exact probabilistic model

```text
Exact Gaussian Process
```

현재처럼

* 1D
* smooth function
* small \(N\)
* kernel prior가 잘 맞는 문제

에서는 GP가 거의 ideal model에 해당한다.

그래서

$$
NLL=0.147
$$

이라는 압도적인 결과가 나왔다.

---

# 17. 가장 중요한 최종 결론

이번 실험은 다음을 굉장히 명확하게 보여준다.

### 1. Prediction accuracy ≠ uncertainty quality

$$
\boxed{
\text{Low RMSE}
\not\Rightarrow
\text{Good uncertainty}
}
$$

MLP가 대표 사례다.

---

### 2. Bayesian model의 장점은 반드시 RMSE 향상이 아니다

Bayesian model이 원하는 것은

$$
p(y|x,D)
$$

전체를 잘 모델링하는 것이다.

즉

```text
Prediction
+
How sure am I?
```

를 함께 맞추는 것이다.

---

### 3. GP는 small-data function regression에서 여전히 매우 강력하다

특히

$$
N\ll1000
$$

이고 kernel prior가 문제에 잘 맞으면 Exact GP는 매우 좋은 baseline이다.

---

### 4. Neural Process의 존재 이유는 GP scalability / amortization이다

GP:

$$
\text{new task}
\Rightarrow
\text{new posterior inference}
$$

NP:

$$
\text{many tasks에서 inference rule 자체를 학습}
$$

한다.

---

### 5. NP가 CNP보다 좋아진 것은 latent variable의 효과를 보여준다

$$
CNP
\rightarrow
r
$$

뿐 아니라

$$
NP
\rightarrow
(r,z)
$$

를 사용하면서 function-level stochasticity를 표현할 수 있었다.

---

# 18. 다음 실험

이 결과에서 바로 이어서 해볼 가장 좋은 ablation은 `n_context` sweep이다.

$$
N_c
=
3,5,10,20,50,100
$$

으로 바꾸고

```text
N_context
 3    5    10    20    50
 │    │     │     │     │
 ▼    ▼     ▼     ▼     ▼

RMSE
│\
│ \
│  \      GP
│   \________
│
│    \       NP
│     \_________
└──────────────── context


NLL
│\
│ \
│  \ GP
│   \__________
│
│      NP
│       \________
└──────────────── context
```

를 그리면 된다.

특히 궁금한 것은

$$
N_c\uparrow
$$

할수록

* GP와 NP의 gap이 줄어드는가?
* BNN calibration이 개선되는가?
* Ensemble diversity가 감소하는가?
* MLP의 coverage가 개선되는가?

이다.

---

# 19. 한 줄 요약

> **MLP와 Ensemble은 값을 잘 맞췄지만 자신감을 너무 과하게 가졌고, GP는 값을 조금 덜 정확하게 맞추더라도 자신이 모르는 영역을 가장 잘 알고 있었다. Neural Process는 GP보다 근사가 거칠었지만, CNP보다 latent function uncertainty를 더 잘 학습했다.**

이번 benchmark의 핵심은 결국 다음 식 하나로 요약된다.

$$
\boxed{
\text{Good Bayesian prediction}
=
\text{accurate mean}
+
\text{well-calibrated uncertainty}
}
$$
