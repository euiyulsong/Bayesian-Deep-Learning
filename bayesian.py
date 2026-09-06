
# -*- coding: utf-8 -*-

"""
run_bayesian_benchmark.py

One-file probabilistic / Bayesian deep-learning benchmark.

Models
------
1) Deterministic MLP
2) MC Dropout
3) Deep Ensemble
4) Mean-field Bayesian Neural Network (Bayes-by-Backprop style VI)
5) Exact Gaussian Process (RBF kernel; analytic posterior)
6) Conditional Neural Process (CNP)
7) Latent Neural Process (NP)

Task
----
1D regression on functions sampled from an RBF Gaussian Process.
The neural-process models are trained meta-learning style across many functions.
The ordinary NN/BNN/ensemble models are trained on ONE target function's context set.
The exact GP uses the same context observations analytically.

Metrics
-------
RMSE
Gaussian predictive NLL
95% predictive interval coverage
Mean predictive interval width
Inference latency

Install
-------
pip install torch numpy pandas matplotlib

Run
---
python run_bayesian_benchmark.py
python run_bayesian_benchmark.py --epochs 1000 --np-epochs 2000
python run_bayesian_benchmark.py --device cuda
python run_bayesian_benchmark.py --quick

Outputs
-------
results/summary.csv
results/predictions.csv
results/comparison.png
results/target_function.png
"""

import argparse
import copy
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import matplotlib.pyplot as plt


# ============================================================
# Reproducibility
# ============================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# Utilities
# ============================================================

LOG_2PI = math.log(2.0 * math.pi)


def gaussian_nll(y, mean, std, eps=1e-6):
    std = np.clip(std, eps, None)
    return float(
        np.mean(0.5 * LOG_2PI + np.log(std) + 0.5 * ((y - mean) / std) ** 2)
    )


def regression_metrics(y, mean, std):
    std = np.clip(std, 1e-6, None)
    rmse = float(np.sqrt(np.mean((y - mean) ** 2)))
    nll = gaussian_nll(y, mean, std)
    lo = mean - 1.96 * std
    hi = mean + 1.96 * std
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    width = float(np.mean(hi - lo))
    return {
        "rmse": rmse,
        "nll": nll,
        "coverage95": coverage,
        "interval_width95": width,
    }


def normal_log_prob(y, mean, std):
    std = torch.clamp(std, min=1e-5)
    return -0.5 * (
        LOG_2PI + 2.0 * torch.log(std) + ((y - mean) / std) ** 2
    )


def timer_ms(fn, repeat=50, warmup=10):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / repeat


# ============================================================
# Synthetic GP function sampler
# ============================================================

def rbf_kernel_np(x1, x2, lengthscale=0.8, variance=1.0):
    x1 = np.asarray(x1).reshape(-1, 1)
    x2 = np.asarray(x2).reshape(-1, 1)
    d2 = (x1 - x2.T) ** 2
    return variance * np.exp(-0.5 * d2 / (lengthscale ** 2))


class GPFunctionGenerator:
    """
    Samples smooth 1-D functions from an RBF Gaussian process.
    """

    def __init__(
        self,
        x_min=-4.0,
        x_max=4.0,
        grid_size=256,
        lengthscale=0.8,
        variance=1.0,
        noise_std=0.10,
        seed=0,
    ):
        self.x_grid = np.linspace(x_min, x_max, grid_size).astype(np.float64)
        self.lengthscale = lengthscale
        self.variance = variance
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)

        K = rbf_kernel_np(
            self.x_grid,
            self.x_grid,
            lengthscale=self.lengthscale,
            variance=self.variance,
        )
        K += 1e-6 * np.eye(grid_size)
        self.L = np.linalg.cholesky(K)

    def sample_function(self):
        z = self.rng.normal(size=len(self.x_grid))
        f = self.L @ z
        return f.astype(np.float32)

    def sample_task(
        self,
        n_context=10,
        n_target=64,
        include_context_in_target=True,
    ):
        f = self.sample_function()

        all_ids = np.arange(len(self.x_grid))
        context_ids = self.rng.choice(all_ids, size=n_context, replace=False)

        if include_context_in_target:
            remaining = np.setdiff1d(all_ids, context_ids)
            extra_n = max(0, n_target - n_context)
            extra_ids = self.rng.choice(
                remaining,
                size=min(extra_n, len(remaining)),
                replace=False
            )
            target_ids = np.concatenate([context_ids, extra_ids])
        else:
            target_ids = self.rng.choice(all_ids, size=n_target, replace=False)

        self.rng.shuffle(target_ids)

        xc = self.x_grid[context_ids].astype(np.float32)[:, None]
        yc_clean = f[context_ids, None]
        yc = yc_clean + self.rng.normal(
            0, self.noise_std, size=yc_clean.shape
        ).astype(np.float32)

        xt = self.x_grid[target_ids].astype(np.float32)[:, None]
        yt_clean = f[target_ids, None]
        yt = yt_clean + self.rng.normal(
            0, self.noise_std, size=yt_clean.shape
        ).astype(np.float32)

        return xc, yc, xt, yt, f[:, None]


# ============================================================
# Ordinary regression networks
# ============================================================

class GaussianMLP(nn.Module):
    """
    Predicts heteroscedastic Normal(mean, std).
    """

    def __init__(self, hidden=128, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.raw_std_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        std = 1e-3 + F.softplus(self.raw_std_head(h))
        return mean, std


def train_gaussian_mlp(
    model,
    x,
    y,
    epochs,
    lr,
    weight_decay=1e-4,
):
    model.train()
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    for _ in range(epochs):
        mean, std = model(x)
        loss = -normal_log_prob(y, mean, std).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

    return model


@torch.no_grad()
def predict_mlp(model, x):
    model.eval()
    mean, std = model(x)
    return mean, std


@torch.no_grad()
def predict_mc_dropout(model, x, samples=100):
    """
    Total predictive variance =
        E[var(y | weights)] + var(E[y | weights])
    """
    model.train()  # activate dropout

    means = []
    vars_aleatoric = []

    for _ in range(samples):
        mean, std = model(x)
        means.append(mean)
        vars_aleatoric.append(std ** 2)

    means = torch.stack(means, dim=0)
    vars_aleatoric = torch.stack(vars_aleatoric, dim=0)

    pred_mean = means.mean(dim=0)
    total_var = vars_aleatoric.mean(dim=0) + means.var(
        dim=0, unbiased=False
    )
    pred_std = torch.sqrt(torch.clamp(total_var, min=1e-8))
    return pred_mean, pred_std


# ============================================================
# Deep ensemble
# ============================================================

def train_ensemble(
    n_models,
    x,
    y,
    epochs,
    lr,
    hidden,
    device,
    base_seed,
):
    models = []

    for i in range(n_models):
        set_seed(base_seed + 100 + i)
        model = GaussianMLP(hidden=hidden, dropout=0.0).to(device)

        # Bootstrap context set to encourage diversity.
        ids = torch.randint(
            low=0,
            high=x.shape[0],
            size=(x.shape[0],),
            device=device,
        )
        xb = x[ids]
        yb = y[ids]

        train_gaussian_mlp(model, xb, yb, epochs, lr)
        models.append(model)

    return models


@torch.no_grad()
def predict_ensemble(models, x):
    means = []
    variances = []

    for model in models:
        model.eval()
        mean, std = model(x)
        means.append(mean)
        variances.append(std ** 2)

    means = torch.stack(means, dim=0)
    variances = torch.stack(variances, dim=0)

    pred_mean = means.mean(dim=0)

    # Law of total variance.
    pred_var = variances.mean(dim=0) + means.var(
        dim=0, unbiased=False
    )
    pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
    return pred_mean, pred_std


# ============================================================
# Mean-field BNN
# ============================================================

class BayesianLinear(nn.Module):
    """
    Mean-field Normal variational posterior:
        q(w) = N(mu, sigma)
    prior:
        p(w) = N(0, prior_std)
    """

    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()

        self.weight_mu = nn.Parameter(
            torch.empty(out_features, in_features).normal_(0, 0.05)
        )
        self.weight_rho = nn.Parameter(
            torch.full((out_features, in_features), -3.0)
        )

        self.bias_mu = nn.Parameter(
            torch.zeros(out_features)
        )
        self.bias_rho = nn.Parameter(
            torch.full((out_features,), -3.0)
        )

        self.prior_std = prior_std

    @staticmethod
    def sigma(rho):
        return F.softplus(rho)

    def sample_params(self):
        w_sigma = self.sigma(self.weight_rho)
        b_sigma = self.sigma(self.bias_rho)

        w = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        return w, b

    def forward(self, x):
        w, b = self.sample_params()
        return F.linear(x, w, b)

    def kl(self):
        prior = Normal(
            torch.tensor(0.0, device=self.weight_mu.device),
            torch.tensor(self.prior_std, device=self.weight_mu.device),
        )

        qw = Normal(self.weight_mu, self.sigma(self.weight_rho))
        qb = Normal(self.bias_mu, self.sigma(self.bias_rho))

        return kl_divergence(qw, prior).sum() + kl_divergence(qb, prior).sum()


class BayesianMLP(nn.Module):
    def __init__(self, hidden=64, prior_std=1.0):
        super().__init__()
        self.l1 = BayesianLinear(1, hidden, prior_std)
        self.l2 = BayesianLinear(hidden, hidden, prior_std)
        self.mean_head = BayesianLinear(hidden, 1, prior_std)
        self.raw_std_head = BayesianLinear(hidden, 1, prior_std)

    def forward(self, x):
        h = torch.tanh(self.l1(x))
        h = torch.tanh(self.l2(h))
        mean = self.mean_head(h)
        std = 1e-3 + F.softplus(self.raw_std_head(h))
        return mean, std

    def kl(self):
        return (
            self.l1.kl()
            + self.l2.kl()
            + self.mean_head.kl()
            + self.raw_std_head.kl()
        )


def train_bnn(
    model,
    x,
    y,
    epochs,
    lr,
    kl_beta=1.0,
    mc_train_samples=3,
):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.shape[0]

    for _ in range(epochs):
        nll = 0.0

        for _ in range(mc_train_samples):
            mean, std = model(x)
            nll = nll - normal_log_prob(y, mean, std).mean()

        nll = nll / mc_train_samples

        # KL/n approximates per-example ELBO.
        kl = model.kl() / max(n, 1)
        loss = nll + kl_beta * kl

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

    return model


@torch.no_grad()
def predict_bnn(model, x, samples=100):
    means = []
    vars_aleatoric = []

    for _ in range(samples):
        mean, std = model(x)
        means.append(mean)
        vars_aleatoric.append(std ** 2)

    means = torch.stack(means)
    vars_aleatoric = torch.stack(vars_aleatoric)

    pred_mean = means.mean(dim=0)
    pred_var = vars_aleatoric.mean(dim=0) + means.var(
        dim=0, unbiased=False
    )
    pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
    return pred_mean, pred_std


# ============================================================
# Exact Gaussian Process
# ============================================================

@dataclass
class ExactGPRegressor:
    lengthscale: float = 0.8
    signal_var: float = 1.0
    noise_std: float = 0.1
    jitter: float = 1e-6

    def kernel(self, x1, x2):
        d2 = (x1[:, None] - x2[None, :]) ** 2
        return self.signal_var * np.exp(
            -0.5 * d2 / (self.lengthscale ** 2)
        )

    def fit(self, x, y):
        self.x_train = np.asarray(x).reshape(-1).astype(np.float64)
        self.y_train = np.asarray(y).reshape(-1).astype(np.float64)

        K = self.kernel(self.x_train, self.x_train)
        K += (
            self.noise_std ** 2 + self.jitter
        ) * np.eye(len(self.x_train))

        self.L = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(
            self.L.T,
            np.linalg.solve(self.L, self.y_train),
        )
        return self

    def predict(self, x, include_noise=True):
        x = np.asarray(x).reshape(-1).astype(np.float64)
        Ks = self.kernel(self.x_train, x)

        mean = Ks.T @ self.alpha

        v = np.linalg.solve(self.L, Ks)
        kss_diag = np.full(
            len(x),
            self.signal_var,
            dtype=np.float64,
        )
        latent_var = kss_diag - np.sum(v ** 2, axis=0)
        latent_var = np.maximum(latent_var, 1e-10)

        pred_var = latent_var.copy()
        if include_noise:
            pred_var += self.noise_std ** 2

        return mean[:, None], np.sqrt(pred_var)[:, None]


# ============================================================
# Conditional Neural Process
# ============================================================

class SetEncoder(nn.Module):
    def __init__(self, hidden=128, r_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, r_dim),
        )

    def forward(self, x, y):
        # x,y: [B,N,1]
        h = self.net(torch.cat([x, y], dim=-1))
        return h


class CNP(nn.Module):
    def __init__(self, hidden=128, r_dim=128):
        super().__init__()
        self.encoder = SetEncoder(hidden, r_dim)
        self.decoder = nn.Sequential(
            nn.Linear(r_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.raw_std_head = nn.Linear(hidden, 1)

    def forward(self, xc, yc, xt):
        r_i = self.encoder(xc, yc)
        r = r_i.mean(dim=1, keepdim=True)
        r = r.expand(-1, xt.shape[1], -1)

        h = self.decoder(torch.cat([xt, r], dim=-1))
        mean = self.mean_head(h)
        std = 1e-3 + F.softplus(self.raw_std_head(h))
        return mean, std


# ============================================================
# Latent Neural Process
# ============================================================

class LatentEncoder(nn.Module):
    def __init__(self, hidden=128, z_dim=64):
        super().__init__()
        self.point = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, z_dim)
        self.raw_std = nn.Linear(hidden, z_dim)

    def forward(self, x, y):
        h = self.point(torch.cat([x, y], dim=-1))
        h = h.mean(dim=1)
        mu = self.mu(h)
        std = 1e-3 + F.softplus(self.raw_std(h))
        return Normal(mu, std)


class NeuralProcess(nn.Module):
    def __init__(self, hidden=128, r_dim=128, z_dim=64):
        super().__init__()

        self.det_encoder = SetEncoder(hidden, r_dim)
        self.latent_encoder = LatentEncoder(hidden, z_dim)

        self.decoder = nn.Sequential(
            nn.Linear(1 + r_dim + z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.raw_std_head = nn.Linear(hidden, 1)

    def deterministic_rep(self, xc, yc, nt):
        r_i = self.det_encoder(xc, yc)
        r = r_i.mean(dim=1, keepdim=True)
        return r.expand(-1, nt, -1)

    def decode(self, xt, r, z):
        z = z[:, None, :].expand(-1, xt.shape[1], -1)
        h = self.decoder(torch.cat([xt, r, z], dim=-1))
        mean = self.mean_head(h)
        std = 1e-3 + F.softplus(self.raw_std_head(h))
        return mean, std

    def forward_train(self, xc, yc, xt, yt):
        # Prior conditioned only on context.
        q_context = self.latent_encoder(xc, yc)

        # Posterior sees targets during training.
        q_target = self.latent_encoder(xt, yt)
        z = q_target.rsample()

        r = self.deterministic_rep(xc, yc, xt.shape[1])
        mean, std = self.decode(xt, r, z)

        kl = kl_divergence(q_target, q_context).sum(dim=-1).mean()
        return mean, std, kl

    def context_distribution(self, xc, yc):
        return self.latent_encoder(xc, yc)

    def predict_sample(self, xc, yc, xt):
        q_context = self.context_distribution(xc, yc)
        z = q_context.rsample()
        r = self.deterministic_rep(xc, yc, xt.shape[1])
        return self.decode(xt, r, z)


# ============================================================
# Meta training helpers
# ============================================================

def make_meta_batch(
    generator,
    batch_size,
    min_context,
    max_context,
    n_target,
    device,
):
    # Same n_context inside one batch to avoid padding.
    nc = np.random.randint(min_context, max_context + 1)

    xcs, ycs, xts, yts = [], [], [], []

    for _ in range(batch_size):
        xc, yc, xt, yt, _ = generator.sample_task(
            n_context=nc,
            n_target=n_target,
            include_context_in_target=True,
        )
        xcs.append(xc)
        ycs.append(yc)
        xts.append(xt)
        yts.append(yt)

    def T(xs):
        return torch.tensor(
            np.stack(xs),
            dtype=torch.float32,
            device=device,
        )

    return T(xcs), T(ycs), T(xts), T(yts)


def train_cnp(
    model,
    generator,
    epochs,
    batch_size,
    min_context,
    max_context,
    n_target,
    lr,
    device,
    print_every=0,
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, epochs + 1):
        xc, yc, xt, yt = make_meta_batch(
            generator,
            batch_size,
            min_context,
            max_context,
            n_target,
            device,
        )

        mean, std = model(xc, yc, xt)
        loss = -normal_log_prob(yt, mean, std).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if print_every and step % print_every == 0:
            print(f"[CNP] step={step:5d} loss={loss.item():.4f}")

    return model


def train_np(
    model,
    generator,
    epochs,
    batch_size,
    min_context,
    max_context,
    n_target,
    lr,
    device,
    kl_weight=1.0,
    print_every=0,
):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(1, epochs + 1):
        xc, yc, xt, yt = make_meta_batch(
            generator,
            batch_size,
            min_context,
            max_context,
            n_target,
            device,
        )

        mean, std, kl = model.forward_train(xc, yc, xt, yt)

        nll = -normal_log_prob(yt, mean, std).mean()
        loss = nll + kl_weight * kl / xt.shape[1]

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        if print_every and step % print_every == 0:
            print(
                f"[NP ] step={step:5d} "
                f"loss={loss.item():.4f} "
                f"nll={nll.item():.4f} "
                f"kl={kl.item():.4f}"
            )

    return model


@torch.no_grad()
def predict_cnp(model, xc, yc, xt):
    model.eval()
    return model(xc, yc, xt)


@torch.no_grad()
def predict_np(model, xc, yc, xt, samples=100):
    model.eval()

    means = []
    variances = []

    for _ in range(samples):
        mean, std = model.predict_sample(xc, yc, xt)
        means.append(mean)
        variances.append(std ** 2)

    means = torch.stack(means)
    variances = torch.stack(variances)

    pred_mean = means.mean(dim=0)
    pred_var = variances.mean(dim=0) + means.var(
        dim=0, unbiased=False
    )
    pred_std = torch.sqrt(torch.clamp(pred_var, min=1e-8))
    return pred_mean, pred_std


# ============================================================
# Evaluation wrapper
# ============================================================

def to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def add_result(
    rows,
    predictions,
    model_name,
    y_true,
    mean,
    std,
    x_eval,
    latency_ms,
):
    y_true = to_np(y_true).reshape(-1)
    mean = to_np(mean).reshape(-1)
    std = to_np(std).reshape(-1)
    x_eval = to_np(x_eval).reshape(-1)

    m = regression_metrics(y_true, mean, std)
    m["model"] = model_name
    m["latency_ms"] = float(latency_ms)
    rows.append(m)

    for x, yt, mu, s in zip(x_eval, y_true, mean, std):
        predictions.append({
            "model": model_name,
            "x": float(x),
            "y_true": float(yt),
            "mean": float(mu),
            "std": float(s),
            "lower95": float(mu - 1.96 * s),
            "upper95": float(mu + 1.96 * s),
        })


# ============================================================
# Plotting
# ============================================================

def plot_target(
    path,
    x_grid,
    f_clean,
    xc,
    yc,
):
    plt.figure(figsize=(10, 5))
    plt.plot(x_grid, f_clean, label="latent function")
    plt.scatter(
        xc.reshape(-1),
        yc.reshape(-1),
        s=35,
        label="context observations",
    )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Target GP Function and Context Set")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(
    path,
    pred_df,
    xc,
    yc,
    max_models=12,
):
    models = list(pred_df["model"].unique())[:max_models]

    ncols = 2
    nrows = math.ceil(len(models) / ncols)

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(13, 4.2 * nrows),
        squeeze=False,
    )

    for ax, model in zip(axes.flatten(), models):
        d = pred_df[pred_df["model"] == model].sort_values("x")

        # Convert pandas Series to NumPy arrays explicitly.
        # This avoids pandas>=2.x / older-matplotlib incompatibility where
        # matplotlib internally tries Series[:, None].
        x = d["x"].to_numpy(dtype=float)
        y_true = d["y_true"].to_numpy(dtype=float)
        mean = d["mean"].to_numpy(dtype=float)
        lower95 = d["lower95"].to_numpy(dtype=float)
        upper95 = d["upper95"].to_numpy(dtype=float)

        ax.plot(x, y_true, label="truth", linewidth=1.5)
        ax.plot(x, mean, label="mean", linewidth=1.5)
        ax.fill_between(
            x,
            lower95,
            upper95,
            alpha=0.25,
            label="95% PI",
        )
        ax.scatter(
            xc.reshape(-1),
            yc.reshape(-1),
            s=18,
            label="context",
        )
        ax.set_title(model)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    for ax in axes.flatten()[len(models):]:
        ax.axis("off")

    handles, labels = axes.flatten()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ============================================================
# Main benchmark
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    p.add_argument("--outdir", type=str, default="results")

    p.add_argument("--n-context", type=int, default=12)
    p.add_argument("--grid-size", type=int, default=256)
    p.add_argument("--noise-std", type=float, default=0.10)
    p.add_argument("--lengthscale", type=float, default=0.8)

    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=1500)
    p.add_argument("--lr", type=float, default=1e-3)

    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--mc-samples", type=int, default=100)

    p.add_argument("--bnn-epochs", type=int, default=2000)
    p.add_argument("--bnn-kl-beta", type=float, default=0.01)

    p.add_argument("--np-epochs", type=int, default=2500)
    p.add_argument("--meta-batch-size", type=int, default=16)
    p.add_argument("--meta-min-context", type=int, default=3)
    p.add_argument("--meta-max-context", type=int, default=30)
    p.add_argument("--meta-targets", type=int, default=64)

    p.add_argument("--quick", action="store_true")
    p.add_argument("--skip-np", action="store_true")
    p.add_argument("--print-every", type=int, default=250)

    return p.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.epochs = 300
        args.bnn_epochs = 400
        args.np_epochs = 500
        args.ensemble_size = 3
        args.mc_samples = 40
        args.meta_batch_size = 8

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 90)
    print("BAYESIAN / PROBABILISTIC DEEP LEARNING BENCHMARK")
    print("=" * 90)
    print("device:", device)
    print("seed:", args.seed)

    # --------------------------------------------------------
    # Separate generators:
    # one for meta-training, one to generate test target.
    # --------------------------------------------------------

    meta_gen = GPFunctionGenerator(
        grid_size=args.grid_size,
        lengthscale=args.lengthscale,
        noise_std=args.noise_std,
        seed=args.seed + 1,
    )

    test_gen = GPFunctionGenerator(
        grid_size=args.grid_size,
        lengthscale=args.lengthscale,
        noise_std=args.noise_std,
        seed=args.seed + 999,
    )

    # One held-out target function.
    xc, yc, _, _, f_full = test_gen.sample_task(
        n_context=args.n_context,
        n_target=args.grid_size,
        include_context_in_target=True,
    )

    x_grid = test_gen.x_grid.astype(np.float32)[:, None]
    y_clean_grid = f_full.astype(np.float32)

    # Evaluate against noisy observations or clean latent?
    # We choose clean function for RMSE while predictive std still
    # includes observation uncertainty for probabilistic metrics.
    y_eval = y_clean_grid

    xt = torch.tensor(x_grid, dtype=torch.float32, device=device)
    yt = torch.tensor(y_eval, dtype=torch.float32, device=device)
    xc_t = torch.tensor(xc, dtype=torch.float32, device=device)
    yc_t = torch.tensor(yc, dtype=torch.float32, device=device)

    rows = []
    predictions = []

    plot_target(
        outdir / "target_function.png",
        x_grid.reshape(-1),
        y_clean_grid.reshape(-1),
        xc,
        yc,
    )

    # --------------------------------------------------------
    # 1. Deterministic Gaussian MLP
    # --------------------------------------------------------
    print("\n[1/7] Deterministic MLP")
    set_seed(args.seed + 10)

    mlp = GaussianMLP(
        hidden=args.hidden,
        dropout=0.0,
    ).to(device)

    train_gaussian_mlp(
        mlp,
        xc_t,
        yc_t,
        epochs=args.epochs,
        lr=args.lr,
    )

    mean, std = predict_mlp(mlp, xt)

    latency = timer_ms(
        lambda: predict_mlp(mlp, xt),
        repeat=30,
        warmup=5,
    )

    add_result(
        rows,
        predictions,
        "MLP",
        yt,
        mean,
        std,
        xt,
        latency,
    )

    # --------------------------------------------------------
    # 2. MC Dropout
    # --------------------------------------------------------
    print("[2/7] MC Dropout")
    set_seed(args.seed + 20)

    mc_model = GaussianMLP(
        hidden=args.hidden,
        dropout=0.10,
    ).to(device)

    train_gaussian_mlp(
        mc_model,
        xc_t,
        yc_t,
        epochs=args.epochs,
        lr=args.lr,
    )

    mean, std = predict_mc_dropout(
        mc_model,
        xt,
        samples=args.mc_samples,
    )

    latency = timer_ms(
        lambda: predict_mc_dropout(
            mc_model,
            xt,
            samples=args.mc_samples,
        ),
        repeat=10,
        warmup=2,
    )

    add_result(
        rows,
        predictions,
        "MC_Dropout",
        yt,
        mean,
        std,
        xt,
        latency,
    )

    # --------------------------------------------------------
    # 3. Deep Ensemble
    # --------------------------------------------------------
    print("[3/7] Deep Ensemble")

    ensemble = train_ensemble(
        n_models=args.ensemble_size,
        x=xc_t,
        y=yc_t,
        epochs=args.epochs,
        lr=args.lr,
        hidden=args.hidden,
        device=device,
        base_seed=args.seed,
    )

    mean, std = predict_ensemble(ensemble, xt)

    latency = timer_ms(
        lambda: predict_ensemble(ensemble, xt),
        repeat=20,
        warmup=3,
    )

    add_result(
        rows,
        predictions,
        f"Deep_Ensemble_{args.ensemble_size}",
        yt,
        mean,
        std,
        xt,
        latency,
    )

    # --------------------------------------------------------
    # 4. Mean-field BNN
    # --------------------------------------------------------
    print("[4/7] Mean-field BNN")

    set_seed(args.seed + 30)
    bnn = BayesianMLP(
        hidden=max(32, args.hidden // 2),
        prior_std=1.0,
    ).to(device)

    train_bnn(
        bnn,
        xc_t,
        yc_t,
        epochs=args.bnn_epochs,
        lr=args.lr,
        kl_beta=args.bnn_kl_beta,
        mc_train_samples=3,
    )

    mean, std = predict_bnn(
        bnn,
        xt,
        samples=args.mc_samples,
    )

    latency = timer_ms(
        lambda: predict_bnn(
            bnn,
            xt,
            samples=args.mc_samples,
        ),
        repeat=10,
        warmup=2,
    )

    add_result(
        rows,
        predictions,
        "BNN_MeanField_VI",
        yt,
        mean,
        std,
        xt,
        latency,
    )

    # --------------------------------------------------------
    # 5. Exact GP
    # --------------------------------------------------------
    print("[5/7] Exact GP")

    gp = ExactGPRegressor(
        lengthscale=args.lengthscale,
        signal_var=1.0,
        noise_std=args.noise_std,
    )
    gp.fit(xc, yc)

    mean_gp, std_gp = gp.predict(
        x_grid,
        include_noise=True,
    )

    latency = timer_ms(
        lambda: gp.predict(x_grid, include_noise=True),
        repeat=100,
        warmup=10,
    )

    add_result(
        rows,
        predictions,
        "Exact_GP",
        y_eval,
        mean_gp,
        std_gp,
        x_grid,
        latency,
    )

    # --------------------------------------------------------
    # 6-7. Neural Processes
    # --------------------------------------------------------
    if not args.skip_np:
        print("[6/7] CNP meta-training")

        set_seed(args.seed + 40)
        cnp = CNP(
            hidden=args.hidden,
            r_dim=args.hidden,
        ).to(device)

        train_cnp(
            cnp,
            meta_gen,
            epochs=args.np_epochs,
            batch_size=args.meta_batch_size,
            min_context=args.meta_min_context,
            max_context=args.meta_max_context,
            n_target=args.meta_targets,
            lr=args.lr,
            device=device,
            print_every=args.print_every,
        )

        xc_b = xc_t[None, ...]
        yc_b = yc_t[None, ...]
        xt_b = xt[None, ...]

        mean, std = predict_cnp(
            cnp,
            xc_b,
            yc_b,
            xt_b,
        )
        mean = mean[0]
        std = std[0]

        latency = timer_ms(
            lambda: predict_cnp(
                cnp,
                xc_b,
                yc_b,
                xt_b,
            ),
            repeat=30,
            warmup=5,
        )

        add_result(
            rows,
            predictions,
            "CNP",
            yt,
            mean,
            std,
            xt,
            latency,
        )

        print("[7/7] Latent Neural Process meta-training")

        # Fresh generator with same distribution but another random stream.
        np_gen = GPFunctionGenerator(
            grid_size=args.grid_size,
            lengthscale=args.lengthscale,
            noise_std=args.noise_std,
            seed=args.seed + 2,
        )

        set_seed(args.seed + 50)
        np_model = NeuralProcess(
            hidden=args.hidden,
            r_dim=args.hidden,
            z_dim=max(16, args.hidden // 2),
        ).to(device)

        train_np(
            np_model,
            np_gen,
            epochs=args.np_epochs,
            batch_size=args.meta_batch_size,
            min_context=args.meta_min_context,
            max_context=args.meta_max_context,
            n_target=args.meta_targets,
            lr=args.lr,
            device=device,
            kl_weight=1.0,
            print_every=args.print_every,
        )

        mean, std = predict_np(
            np_model,
            xc_b,
            yc_b,
            xt_b,
            samples=args.mc_samples,
        )
        mean = mean[0]
        std = std[0]

        latency = timer_ms(
            lambda: predict_np(
                np_model,
                xc_b,
                yc_b,
                xt_b,
                samples=args.mc_samples,
            ),
            repeat=10,
            warmup=2,
        )

        add_result(
            rows,
            predictions,
            "Neural_Process",
            yt,
            mean,
            std,
            xt,
            latency,
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------
    summary = pd.DataFrame(rows)
    summary = summary[
        [
            "model",
            "rmse",
            "nll",
            "coverage95",
            "interval_width95",
            "latency_ms",
        ]
    ].sort_values("nll")

    pred_df = pd.DataFrame(predictions)

    summary.to_csv(outdir / "summary.csv", index=False)
    pred_df.to_csv(outdir / "predictions.csv", index=False)

    plot_comparison(
        outdir / "comparison.png",
        pred_df,
        xc,
        yc,
    )

    print()
    print("=" * 100)
    print("RESULT")
    print("=" * 100)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nSaved:")
    print(" ", outdir / "summary.csv")
    print(" ", outdir / "predictions.csv")
    print(" ", outdir / "target_function.png")
    print(" ", outdir / "comparison.png")


if __name__ == "__main__":
    main()
