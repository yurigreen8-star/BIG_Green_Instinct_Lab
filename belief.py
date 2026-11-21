"""
belief.py — байесовское обновление для гауссовской модели.

Часть BIG_Green Instinct Lab, v0.1.
Вариант A: один латентный скаляр качества q_i на участок.

Модель:
    Приор:          q_i ~ N(mu_i, sigma2_i)
    Правдоподобие:  o   ~ N(q_i, s^2)
    Апостериор:     через сопряжённое обновление по точностям.

Инварианты:
    - known[i] == True  <=>  sigma2[i] == 0
    - Малая положительная дисперсия (например, 1e-10) НЕ делает known.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

# Числовой порог: после обновления дисперсия ниже этого значения
# логируется как предупреждение, но НЕ приравнивается к 0.
EPS_NUM = 1e-12


@dataclass
class Belief:
    """
    Апостериорное распределение качеств участков.

    Атрибуты:
        mu:     массив средних,     форма (n,)
        sigma2: массив дисперсий,   форма (n,), >= 0
        known:  булев массив,       форма (n,)
    """

    mu: np.ndarray
    sigma2: np.ndarray
    known: np.ndarray

    def __post_init__(self) -> None:
        self.mu = np.asarray(self.mu, dtype=np.float64)
        self.sigma2 = np.asarray(self.sigma2, dtype=np.float64)
        self.known = np.asarray(self.known, dtype=bool)

        if not (self.mu.shape == self.sigma2.shape == self.known.shape):
            raise ValueError(
                "mu, sigma2, known должны иметь одинаковую форму"
            )
        if self.mu.ndim != 1:
            raise ValueError("mu должен быть одномерным")
        if np.any(self.sigma2 < 0):
            raise ValueError("sigma2 не может быть отрицательной")
        # Проверка инварианта: known <=> sigma2 == 0
        if not np.array_equal(self.known, self.sigma2 == 0.0):
            raise ValueError(
                "known должен быть эквивалентен (sigma2 == 0)"
            )

    @property
    def n(self) -> int:
        return self.mu.shape[0]

    def copy(self) -> "Belief":
        return Belief(
            mu=self.mu.copy(),
            sigma2=self.sigma2.copy(),
            known=self.known.copy(),
        )

    def update(
        self, parcel_id: int, observation: float, noise_var: float
    ) -> "Belief":
        """
        Байесовское обновление для гауссовского сопряжения.

        Возвращает НОВЫЙ Belief (иммутабельность).

        Случаи:
            1. s^2 > 0, sigma^2 > 0 : стандартная формула через точности
            2. s^2 = 0              : точное измерение -> known = True
            3. sigma^2 = 0 (уже known): ValueError (повторный зонд)
            4. sigma'^2 < EPS_NUM   : предупреждение, НЕ приравнивается к 0
        """
        if noise_var < 0:
            raise ValueError(
                f"noise_var должен быть >= 0, получено {noise_var}"
            )
        if not (0 <= parcel_id < self.n):
            raise ValueError(f"parcel_id вне диапазона: {parcel_id}")

        # Случай 3: попытка зондировать уже известный участок.
        if self.known[parcel_id]:
            raise ValueError(
                f"Участок {parcel_id} уже known. "
                f"Повторное зондирование запрещено."
            )

        new_mu = self.mu.copy()
        new_sigma2 = self.sigma2.copy()
        new_known = self.known.copy()

        if noise_var == 0.0:
            # Случай 2: точное измерение.
            new_mu[parcel_id] = observation
            new_sigma2[parcel_id] = 0.0
            new_known[parcel_id] = True
        else:
            # Случай 1: гауссовское сопряжение через точности.
            tau = 1.0 / self.sigma2[parcel_id]
            tau_o = 1.0 / noise_var
            tau_new = tau + tau_o
            new_mu[parcel_id] = (
                tau * self.mu[parcel_id] + tau_o * observation
            ) / tau_new
            new_sigma2[parcel_id] = 1.0 / tau_new

            # Случай 4: численно малая дисперсия.
            if new_sigma2[parcel_id] < EPS_NUM:
                warnings.warn(
                    f"Дисперсия участка {parcel_id} после обновления "
                    f"({new_sigma2[parcel_id]:.2e}) ниже EPS_NUM "
                    f"({EPS_NUM:.2e}). Участок НЕ помечается как known.",
                    stacklevel=2,
                )

        return Belief(mu=new_mu, sigma2=new_sigma2, known=new_known)