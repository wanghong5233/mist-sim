"""P3 monotone roots and a Euclidean projection, in float64."""
import numpy as np


def solve_p3(alpha, beta, minimum, capacity, tolerance=1e-10):
    alpha, beta, minimum = (np.asarray(values, dtype=np.float64) for values in (alpha, beta, minimum))
    if alpha.shape != beta.shape or beta.shape != minimum.shape or alpha.ndim != 1:
        raise ValueError("P3 shape")
    if not all(np.isfinite(values).all() for values in (alpha, beta, minimum)):
        raise ValueError("Nonfinite P3")
    if np.any(alpha <= 0) or np.any(beta < 0) or np.any(minimum <= 0) or not np.isfinite(capacity) or capacity < 0:
        raise ValueError("Invalid P3 domain")
    if minimum.sum() > capacity + tolerance:
        raise ValueError("Infeasible P3 capacity")
    if not len(minimum) or capacity - minimum.sum() <= tolerance:
        return minimum.copy()
    unconstrained = np.maximum(minimum, np.cbrt(beta / (2 * alpha)))
    if unconstrained.sum() <= capacity:
        return unconstrained

    def rates(multiplier):
        lower, upper = minimum.copy(), unconstrained.copy()
        current = upper.copy()
        clamped = 2 * alpha * minimum - beta / minimum ** 2 + multiplier >= 0
        for _ in range(55):
            derivative = 2 * alpha * current - beta / current ** 2 + multiplier
            scale = 2 * alpha * current + beta / current ** 2 + multiplier
            converged = clamped | (np.abs(derivative) <= 1e-13 * scale)
            if np.all(converged):
                return np.where(clamped, minimum, current)
            lower = np.where((derivative < 0) & ~converged, current, lower)
            upper = np.where((derivative >= 0) & ~converged, current, upper)
            proposal = current - derivative / (2 * alpha + 2 * beta / current ** 3)
            proposal = np.where((proposal > lower) & (proposal < upper), proposal, (lower + upper) / 2)
            current = np.where(converged, current, proposal)
        return np.where(clamped, minimum, current)

    low, high = 0.0, float(np.max(np.maximum(0, beta / minimum ** 2 - 2 * alpha * minimum))) + 1
    feasible = minimum.copy()
    for _ in range(65):
        middle = (low + high) / 2
        trial = rates(middle)
        total = trial.sum()
        if total > capacity:
            low = middle
        else:
            high = middle
            feasible = trial
            if capacity - total <= tolerance:
                break
    return feasible


def project_rates(predicted, minimum, capacity):
    predicted, minimum = np.asarray(predicted, dtype=float), np.asarray(minimum, dtype=float)
    if (predicted.shape != minimum.shape or predicted.ndim != 1 or not np.isfinite(predicted).all()
            or not np.isfinite(minimum).all() or np.any(minimum <= 0) or not np.isfinite(capacity) or capacity < 0):
        raise ValueError("Invalid predicted rates")
    if np.sum(minimum) > capacity + 1e-9:
        raise ValueError("Infeasible projection")
    clipped = np.maximum(predicted, minimum)
    if clipped.sum() <= capacity:
        return clipped
    low, high = 0.0, max(0.0, float(np.max(predicted - minimum)))
    for _ in range(70):
        middle = (low + high) / 2
        if np.maximum(minimum, predicted - middle).sum() > capacity:
            low = middle
        else:
            high = middle
    return np.maximum(minimum, predicted - high)
