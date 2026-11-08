"""Experimentos reproducibles del problema 1: anillos y clase seno."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mpmath as mp
import numpy as np


OUT = Path(__file__).parent / "resultados"


def ring_count_for_ordered_labels(y: np.ndarray) -> int:
    """Cuenta etiquetas realizables por un intervalo radial, incluyendo vacío."""
    y = np.asarray(y, dtype=np.int8)
    positives = np.flatnonzero(y == 1)
    
    if positives.size == 0:
        return 1
    return int(np.sum(positives[:, None] <= positives[None, :]))


def realizable(points: np.ndarray, labels: np.ndarray) -> bool:
    """Determina si un anillo centrado en el origen produce estas etiquetas."""
    points = np.asarray(points, dtype=float)
    labels = np.asarray(labels, dtype=int)
    
    if points.ndim != 2 or points.shape[1] != 2 or labels.shape != (len(points),):
        raise ValueError("Se esperan puntos de forma (n, 2) y n etiquetas.")
    
    if not np.isin(labels, (-1, 1)).all():
        raise ValueError("Las etiquetas deben ser -1 o +1.")
    
    order = np.argsort(np.linalg.norm(points, axis=1), kind="stable")
    y = labels[order]
    pos = np.flatnonzero(y == 1)

    
    return pos.size == 0 or pos[-1] - pos[0] + 1 == pos.size


def experiment_a(rng: np.random.Generator) -> dict:
    ns = np.arange(1, 13)
    empirical = []
    for n in ns:

        best = 0
        for _ in range(100):
            points = rng.uniform(-1, 1, size=(n, 2))
            radii = np.linalg.norm(points, axis=1)
            order = np.argsort(radii, kind="stable")
            count = 0
            for mask in range(1 << n):
                labels = np.array([1 if (mask >> k) & 1 else -1 for k in range(n)])
                if realizable(points, labels):
                    count += 1

            best = max(best, count)
        empirical.append(best)

    empirical = np.array(empirical)
    exact = 1 + ns * (ns + 1) // 2
    full = 2.0**ns
    sauer = (ns + 1.0) ** 2


    with (OUT / "conteo_etiquetaciones.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "conteo_empirico_maximo", "formula_exacta", "2^n", "sauer_d2"])
        w.writerows(zip(ns, empirical, exact, full, sauer))
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(ns, empirical, "o-", label="Máximo empírico (100 acomodos)")
    ax.plot(ns, exact, "s--", label=r"$m_H(n)=1+n(n+1)/2$")
    ax.plot(ns, full, "^-", label=r"$2^n$")
    ax.plot(ns, sauer, "d-", label=r"Sauer, $d=2$: $(n+1)^2$")
    ax.set_yscale("log")
    ax.set_xlabel("n")
    ax.set_ylabel("Número de etiquetaciones (escala logarítmica)")
    ax.set_title("Conteo de etiquetaciones para anillos centrados en el origen")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "conteo_etiquetaciones.png", dpi=180)
    plt.close(fig)
    
    return {"n": ns.tolist(), "empirical": empirical.tolist(), "exact": exact.tolist()}


def erm_ring(points: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    """ERM exacto O(n log n): busca el mejor bloque positivo radial."""
    r = np.linalg.norm(points, axis=1)
    order = np.argsort(r, kind="stable")
    rs, ys = r[order], labels[order]
    # Con anillos cerrados, los empates radiales deben recibir la misma etiqueta

    # En la distribuciopn continua usada aquí los empates tienen probabilidad cero
    delta = np.where(ys == -1, 1, -1)
    prefix = np.concatenate(([0], np.cumsum(delta)))
    positives = int(np.sum(ys == 1))
    best_loss = positives  # anillo vacío
    best_i, best_j = -1, -1
    max_prefix, max_i = 0, 0
    min_i = 0
    for j in range(1, len(ys) + 1):
        gain_cost = int(prefix[j] - max_prefix)
        loss = positives + gain_cost
        if loss < best_loss:
            best_loss, best_i, best_j = loss, min_i, j - 1
        if prefix[j] > max_prefix:
            max_prefix, max_i = int(prefix[j]), j
            min_i = max_i
    if best_i < 0:
        return math.inf, math.inf, best_loss / len(ys)
    return float(rs[best_i]), float(rs[best_j]), best_loss / len(ys)


def ring_predict(points: np.ndarray, r1: float, r2: float) -> np.ndarray:

    if math.isinf(r1):
        return -np.ones(len(points), dtype=int)
    r = np.linalg.norm(points, axis=1)

    return np.where((r >= r1) & (r <= r2), 1, -1)


def noisy_ring_data(rng: np.random.Generator, n: int, noise: float) -> tuple[np.ndarray, np.ndarray]:

    x = rng.uniform(-1, 1, size=(n, 2))
    r = np.linalg.norm(x, axis=1)
    y = np.where((r >= 0.30) & (r <= 0.70), 1, -1)
    flip = rng.random(n) < noise
    y[flip] *= -1

    return x, y


def experiment_b(rng: np.random.Generator) -> dict:
    ns = np.array([10, 20, 50, 100, 200, 500, 1000, 2000])
    reps, delta, noise = 30, 0.05, 0.10
    rows = []
    means_in, means_out, bounds = [], [], []

    for n in ns:
        ein_rep, eout_rep = [], []
        for _ in range(reps):
            x, y = noisy_ring_data(rng, int(n), noise)
            r1, r2, ein = erm_ring(x, y)
            xt, yt = noisy_ring_data(rng, 10_000, noise)
            eout = float(np.mean(ring_predict(xt, r1, r2) != yt))
            ein_rep.append(ein)
            eout_rep.append(eout)
        ein_mean = float(np.mean(ein_rep))
        eout_mean = float(np.mean(eout_rep))
        bound = math.sqrt(8 / n * math.log(4 * (2 * n + 1) ** 2 / delta))
        rows.append([n, reps, ein_mean, eout_mean, eout_mean - ein_mean, bound])
        means_in.append(ein_mean)
        means_out.append(eout_mean)
        bounds.append(bound)
    with (OUT / "curva_aprendizaje.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["n", "repeticiones", "Ein_promedio", "Eout_promedio", "brecha_promedio", "cota_P1_Sauer"])
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(ns, np.array(means_out) - means_in, "o-", label=r"Promedio $E_{out}-E_{in}$")
    ax.plot(ns, bounds, "s--", label="Cota P1 + Sauer, d=2")
    ax.set_xscale("log")
    ax.set_xlabel("Tamaño de muestra n")
    ax.set_ylabel("Brecha de error")
    ax.set_title("Curva de aprendizaje y cota teórica")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "curva_aprendizaje.png", dpi=180)
    plt.close(fig)
    return {"rows": rows, "truth_ring": [0.30, 0.70], "noise": noise, "repetitions": reps}


def sine_params(labels: list[int]) -> int:

    return 1 + sum(((1 - y) // 2) * 10**i for i, y in enumerate(labels, start=1))


def experiment_c_i(rng: np.random.Generator) -> dict:
    rows = []

    for m in range(1, 26):

        float_ok = 0
        mp_ok = 0

        for _ in range(200):
            labels = rng.choice([-1, 1], size=m).tolist()
            omega = math.pi * float(sine_params(labels))
            float_predictions = []
            for i, y in enumerate(labels, start=1):
                xi = 10.0 ** (-i)
                pred = 1 if math.sin(omega * xi) >= 0 else -1
                float_predictions.append(pred)
            float_ok += int(float_predictions == labels)
            mp.mp.dps = max(100, 2 * m + 50)
            wmp = mp.pi * sine_params(labels)
            mp_predictions = []
            for i, y in enumerate(labels, start=1):
                xi = mp.power(10, -i)
                pred = 1 if mp.sin(wmp * xi) >= 0 else -1
                mp_predictions.append(pred)
            mp_ok += int(mp_predictions == labels)
        rows.append([m, 200, float_ok / 200, mp_ok / 200])
    with (OUT / "seno_precision.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["m", "etiquetaciones", "fraccion_etiquetaciones_float64_correctas", "fraccion_etiquetaciones_mpmath_correctas"])
        w.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([r[0] for r in rows], [r[2] for r in rows], "o-", label="float64")
    ax.plot([r[0] for r in rows], [r[3] for r in rows], "s--", label="mpmath")
    ax.axhline(1, color="black", linewidth=0.8)
    ax.set_xlabel("Número de puntos m")
    ax.set_ylabel("Fracción de etiquetas reproducidas")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Realización de etiquetas por la clase seno")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "seno_precision.png", dpi=180)
    plt.close(fig)
    return {"rows": rows}


def sine_predict(x: np.ndarray, omega: float) -> np.ndarray:

    return np.where(np.sin(omega * x) >= 0, 1, -1)


def experiment_c_ii(rng: np.random.Generator) -> dict:
    As = [30, 300, 3000, 30000]
    ns = [20, 50, 200]
    grid_size = 60_000
    noise = 0.15
    datasets = {}

    for n in ns:
        x = rng.random(n)
        y = np.where(np.sin(6 * np.pi * x) >= 0, 1, -1)
        y[rng.random(n) < noise] *= -1
        xt = rng.random(10_000)
        yt = np.where(np.sin(6 * np.pi * xt) >= 0, 1, -1)
        yt[rng.random(10_000) < noise] *= -1
        datasets[n] = (x, y, xt, yt)
    rows = []
    
    for A in As:
        omegas = np.linspace(A / grid_size, A, grid_size)
        for n in ns:
            x, y, xt, yt = datasets[n]
            best_ein, best_omega = 1.0, float(omegas[0])
            for start in range(0, grid_size, 512):
                ws = omegas[start : start + 512]
                pred = np.where(np.sin(ws[:, None] * x[None, :]) >= 0, 1, -1)
                errs = np.mean(pred != y[None, :], axis=1)
                idx = int(np.argmin(errs))
                if float(errs[idx]) < best_ein:
                    best_ein, best_omega = float(errs[idx]), float(ws[idx])
            eout = float(np.mean(sine_predict(xt, best_omega) != yt))
            rows.append([A, n, grid_size, best_omega, best_ein, eout])
    with (OUT / "seno_busqueda_omega.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["A", "n", "puntos_rejilla", "omega_elegido", "Ein", "Eout"])
        w.writerows(rows)
    return {"rows": rows, "noise": noise, "grid_size": grid_size}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True, help="Expediente completo usado como semilla")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(args.seed)
    results = {
        "seed": args.seed,
        "a": args.seed % 10,
        "b": (args.seed // 10) % 10,
        "inciso_a": experiment_a(rng),
        "inciso_b": experiment_b(rng),
        "inciso_c_i": experiment_c_i(rng),
        "inciso_c_ii": experiment_c_ii(rng),
    }
    (OUT / "resumen.json").write_text(
        json.dumps(results, indent=2, default=lambda value: value.item() if isinstance(value, np.generic) else str(value)),
        encoding="utf-8",
    )
    print(f"Resultados guardados en {OUT}")


if __name__ == "__main__":
    main()
