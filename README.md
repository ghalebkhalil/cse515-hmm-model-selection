# cse515-hmm-model-selection

Likelihood-Based Model Selection in High-Dimensional Hidden Markov Models  
**CSE 515 Project — University of Washington**  
**Author:** Ghaleb Khalil

## Overview

This project develops a **Hidden Markov Model (HMM) from scratch** for likelihood-based model selection in high-dimensional time-series settings. The implementation focuses on **Gaussian-emission HMMs**, with an emphasis on:

- numerically stable likelihood computation
- EM estimation
- AIC/BIC-based model selection
- state alignment across fits
- redundancy detection
- instability diagnostics across restarts

The main goal is not just to fit an HMM, but to study whether classical likelihood criteria such as **AIC** and **BIC** remain reliable in **high-dimensional settings**, and to supplement them with **stability and redundancy diagnostics**.

---

## Model Setup

The model is a finite-state Gaussian-emission HMM:

- Hidden states: $Z_t \in \{1, \dots, K\}$
- Observations: $X_t \in \mathbb{R}^d$
- Parameters:
  - initial distribution $\pi$
  - transition matrix $A$
  - state means $\mu_k$
  - state covariance matrices $\Sigma_k$, for $k = 1, \dots, K$

At each time step, the hidden state evolves according to the Markov transition matrix, and the observation is generated from a Gaussian distribution associated with that hidden state.

---

## Main Contributions

This repository implements the full HMM pipeline from scratch, including:

- **Parameter validation**
- **Gaussian emission log-density computation**
- **Scaled forward filtering**
- **Backward recursion**
- **Smoothing probabilities**
- **EM / Baum–Welch estimation**
- **Multiple random restarts**
- **AIC/BIC parameter counting**
- **Model comparison over a grid of hidden states**
- **Hungarian alignment of latent states across fits**
- **Symmetric KL-based redundancy diagnostics**
- **Instability metrics across fitted models**

This is a custom implementation designed for research and model-diagnostics work, not a wrapper around an external HMM package.

---

## Repository Logic

The code is organized around the following phases.

### Phase 0 — Define the first working HMM

Start with the Gaussian-emission HMM:

- hidden states $Z_t \in \{1, \dots, K\}$
- observations $X_t \in \mathbb{R}^d$
- parameters $(\pi, A, \mu_k, \Sigma_k)$

This creates the foundation for likelihood computation, inference, and model selection.

---

### Phase 1 — Scaled Forward Filter + Log-Likelihood

This is the most important core block in the project.

For each time $t$:

**Prediction**

$$
\hat{\phi}_t(j) = \sum_i \phi_{t-1}(i) A_{ij}
$$

**Update and scaling**

$$
c_t = \sum_j \hat{\phi}_t(j)\, g_j(x_t), \qquad
\phi_t(j) = \frac{\hat{\phi}_t(j)\, g_j(x_t)}{c_t}
$$

**Log-likelihood**

$$
\ell(\theta) = \sum_{t=1}^T \log c_t
$$

Why this matters:

- avoids numerical underflow
- gives the log-likelihood directly
- provides the backbone for EM
- is required for AIC/BIC

---

### Phase 2 — Backward Recursion + Smoothing

Once the forward recursion is stable, the next step is backward recursion and smoothing.

The implementation computes:

- smoothed state probabilities

$$
\gamma_t(k) = P(Z_t = k \mid x_{1:T})
$$

- smoothed pair probabilities

$$
\xi_t(i,j) = P(Z_t = i, Z_{t+1} = j \mid x_{1:T})
$$

These are the sufficient statistics needed for the EM updates.

---

### Phase 3 — EM (Baum–Welch) for Gaussian Emissions

Given $\gamma$ and $\xi$, the code updates:

**Initial probabilities**

$$
\pi_k \leftarrow \gamma_1(k)
$$

**Transition probabilities**

$$
A_{ij} \leftarrow
\frac{\sum_{t=1}^{T-1} \xi_t(i,j)}
{\sum_{t=1}^{T-1} \gamma_t(i)}
$$

**Means**

$$
\mu_k \leftarrow
\frac{\sum_t \gamma_t(k)x_t}
{\sum_t \gamma_t(k)}
$$

**Covariances**

$$
\Sigma_k \leftarrow
\frac{\sum_t \gamma_t(k)(x_t-\mu_k)(x_t-\mu_k)^\top}
{\sum_t \gamma_t(k)} + \varepsilon I
$$

The code supports:

- **full covariance**
- **diagonal covariance**

Stopping criteria:

- log-likelihood improvement below tolerance
- or maximum iteration count

To reduce sensitivity to initialization, the code supports **multiple random restarts** and keeps the fit with the best final log-likelihood.

---

### Phase 4 — AIC/BIC Model Selection

For each candidate number of states $K$, the project computes:

- log-likelihood
- number of free parameters
- AIC
- BIC

The formulas used are:

$$
\mathrm{AIC} = -2\ell + 2p
$$

$$
\mathrm{BIC} = -2\ell + p \log T
$$

where:

- $\ell$ is the fitted log-likelihood
- $p$ is the number of free parameters
- $T$ is the sample size

The parameter counting includes:

- $K - 1$ parameters for $\pi$
- $K(K - 1)$ for $A$
- $Kd$ for the means
- covariance parameters depending on whether the covariance is full or diagonal

This creates the main model-selection pipeline over a grid of state counts.

---

### Phase 5 — Stability and Redundancy Diagnostics

This is the part that goes beyond simply fitting an HMM.

#### A. State Alignment via Hungarian Algorithm

Because latent states are only identifiable up to permutation, different fits can represent the same model with different state labels.

The code aligns states across fits using a cost matrix based on either:

- Euclidean distance between means
- symmetric KL divergence between Gaussian emissions

Then it solves the assignment problem using the **Hungarian algorithm**.

#### B. Redundancy Diagnostics

To detect potentially redundant states, the code computes the KL divergence between Gaussian emissions.

For two Gaussian components:

$$
\mathrm{KL}(\mathcal{N}_0 \,\|\, \mathcal{N}_1)
=
\frac{1}{2}
\left(
\mathrm{tr}(\Sigma_1^{-1}\Sigma_0)
+
(\mu_1-\mu_0)^\top \Sigma_1^{-1} (\mu_1-\mu_0)
- d
+
\log\frac{\det \Sigma_1}{\det \Sigma_0}
\right)
$$

Then it forms the **symmetric KL divergence**:

$$
\mathrm{symKL}(p,q)
=
\frac{1}{2}\big(\mathrm{KL}(p \| q) + \mathrm{KL}(q \| p)\big)
$$

Low symmetric KL, similar transition rows, and small state occupancy can indicate a redundant hidden state.

#### C. Instability Metrics

After alignment, the code compares two fitted models using:

- Frobenius norm of transition matrices

$$
\|A_1 - A_2\|_F
$$

- mean symmetric KL between aligned emissions
- disagreement rate between MAP state paths

These diagnostics help assess whether a selected value of $K$ is genuinely stable or only appears favorable by likelihood criteria.

---

## Functions Included

### Core Validation and Emission Functions

- `hmm_parameters(...)`  
  Validates shapes, probability constraints, symmetry, and positive-definiteness.

- `prepare_gaussian_cache(...)`  
  Precomputes Cholesky factors and log-determinants for Gaussian emissions.

- `compute_logB(...)`  
  Computes the log emission density matrix $\log B$.

---

### Forward / Backward / Smoothing

- `forward_filter_from_logB(...)`  
  Runs the scaled forward filter and returns filtered probabilities and log-likelihood.

- `backward_normalized_from_logB(...)`  
  Runs the normalized backward recursion.

- `smooth_gamma_xi_fast(...)`  
  Computes filtered probabilities, backward quantities, smoothing probabilities, pair probabilities, and log-likelihood.

---

### EM Estimation

- `m_step_updates(...)`  
  Performs the M-step for Gaussian HMMs.

- `baum_welch_em_fast(...)`  
  Runs EM until convergence.

---

### Initialization and Restarts

- `_random_stochastic_matrix(...)`
- `_init_gaussian_params_from_data(...)`
- `fit_best_of_restarts_fast(...)`

These functions generate initial parameters and run multiple restarts, keeping the best fit.

---

### Model-Selection Utilities

- `num_params(...)`
- `aic(...)`
- `bic(...)`
- `score_over_K_fast(...)`

These support model fitting over a grid of $K$ values.

---

### Alignment, Redundancy, and Instability

- `_hungarian_min_cost(...)`
- `cost_matrix_mu_l2(...)`
- `cost_matrix_emission_symkl(...)`
- `apply_permutation_to_model(...)`
- `align_model_to_reference(...)`
- `pairwise_symkl_within_model(...)`
- `redundancy_report(...)`
- `frobenius_A(...)`
- `mean_state_symkl(...)`
- `map_path_disagreement(...)`
- `compare_models_instability(...)`
- `instability_against_reference(...)`

---
