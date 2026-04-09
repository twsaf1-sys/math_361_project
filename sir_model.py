"""
SIR model for COVID-19 on a university campus (Morgan State–style)
with parameter sweeps on:
  - contact rate (transmission opportunity),
  - infection probability per contact,
  - recovery rate,
  - initial infected I0.

For EACH combination, this script (in parallel, via multithreading):
  - Simulates the SIR model for 120 days (dt = 0.25),
  - Saves a time-series CSV,
  - Saves an SIR plot (PNG, dpi=400, tight layout),
  - Optionally saves a heatmap GIF animation (S/I/R), where each cell = one individual.
  - After all simulations, builds:
      * β–γ heatmaps of peak infections (absolute) for each I0
      * β–γ heatmaps of peak infections (fraction of population) for each I0

Parameter sweeps:
  contact_scale, infection_scale, gamma_scale ∈ {0.90, 0.95, 1.00, 1.05, 1.10}
  I0 ∈ {1, 2, 5, 10, 100, 500}

Total scenarios:
  5 (contact scales) * 5 (infection scales)
  * 5 (recovery scales) * 6 (I0 values) = 750.

File naming:
  CSV:
    sir_<number_initially_infected>_initial_infection_<transmission_rate>_<infection_rate>_<recovery_rate>.csv
  PNG (time series):
    sir_<number_initially_infected>_initial_infection_<transmission_rate>_<infection_rate>_<recovery_rate>_timeseries.png
  GIF (heatmap, optional):
    sir_<number_initially_infected>_initial_infection_<transmission_rate>_<infection_rate>_<recovery_rate>_heatmap.gif
  PNG (β–γ peak infection heatmaps, one per I0, absolute):
    sir_heatmap_peakI_I0_<I0>.png
  PNG (β–γ peak infection heatmaps, one per I0, fraction):
    sir_heatmap_peakI_fraction_I0_<I0>.png

Parallelization:
  Uses concurrent.futures.ThreadPoolExecutor (multithreading).
  Scenarios are processed in batches of 50. If the total number of
  scenarios is not divisible by 50, the final batch is the smaller
  remainder batch.

  Each batch uses up to 24 worker threads (or fewer if the batch is smaller).

Timing:
  - Records total wall-clock time.
  - Records per-scenario compute time.
  - Reports total scenario compute time (sum of per-scenario times),
    which is an approximate "core-seconds" measure of work done.
"""

import os
import time
import itertools
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # important for headless / non-GUI environments
import matplotlib.pyplot as plt

from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from matplotlib import animation

# ---------------------------------------------------------
# Optional psutil import (not used right now, but kept handy)
# ---------------------------------------------------------
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ---------------------------------------------------------
# Global lock for matplotlib operations (not fully thread-safe)
# ---------------------------------------------------------
MATPLOTLIB_LOCK = threading.Lock()

# ---------------------------------------------------------
# 0. Output folders
# ---------------------------------------------------------

DATA_DIR = "sir_data"
PLOT_DIR = "sir_plots"
ANIM_DIR = "sir_heatmaps"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(ANIM_DIR, exist_ok=True)

# Toggle GIF creation here: False = skip GIFs (default), True = make GIFs
MAKE_ANIMATIONS = False

# ---------------------------------------------------------
# 1. Campus population and baseline parameters
# ---------------------------------------------------------

# Morgan State numbers (Fall 2024)
TOTAL_STUDENTS = 10_739
TOTAL_FACULTY = 613
N = TOTAL_STUDENTS + TOTAL_FACULTY  # 11,352

# Sweep over these initial infected counts
INITIAL_INFECTED_VALUES = [1, 2, 5, 10, 100, 500]
R0_initial = 0  # recovered at t = 0

# Time grid: 120 days, quarter-day increments
T_MAX = 120.0
DT = 0.25
t = np.arange(0.0, T_MAX + DT, DT, dtype=float)

# COVID baseline parameters
R0_COVID = 3.32          # approx basic reproduction number
gamma0 = 1.0 / 8.0       # recovery rate per day (~8-day infectious period)
beta0 = R0_COVID * gamma0

# Split beta into contact rate and infection probability
contact_rate0 = 10.0     # close contacts per person per day
infection_prob0 = beta0 / contact_rate0

print("=== Baseline parameters ===")
print(f"N = {N}")
print(f"Baseline R0      = {R0_COVID:.3f}")
print(f"Baseline gamma   = {gamma0:.4f} per day")
print(f"Baseline beta    = {beta0:.4f} per day")
print(f"Baseline contact = {contact_rate0:.2f} contacts/day")
print(f"Baseline p_inf   = {infection_prob0:.4f} per contact")
print()

# ---------------------------------------------------------
# 2. Utility functions
# ---------------------------------------------------------

def fmt_scale(scale: float) -> str:
    """Format a scale like 0.90 as '090' so filenames are clean."""
    return f"{int(round(scale * 100)):03d}"

def run_sir_simulation(beta, gamma, N, S0, I0, R0_initial, t_array):
    """Simple forward-Euler SIR integration."""
    dt = t_array[1] - t_array[0]
    S = np.zeros_like(t_array, dtype=float)
    I = np.zeros_like(t_array, dtype=float)
    R = np.zeros_like(t_array, dtype=float)

    S[0], I[0], R[0] = float(S0), float(I0), float(R0_initial)

    for k in range(len(t_array) - 1):
        S_k, I_k, R_k = S[k], I[k], R[k]

        dS = -beta * S_k * I_k / N
        dI = beta * S_k * I_k / N - gamma * I_k
        dR = gamma * I_k

        S[k + 1] = S_k + dt * dS
        I[k + 1] = I_k + dt * dI
        R[k + 1] = R_k + dt * dR

    return S, I, R

# ---------- NEW: analytical equilibrium helper ----------

def compute_equilibrium_points(beta, gamma, N):
    """
    Compute analytical equilibrium information for the closed SIR system:

        dS/dt = -β S I / N
        dI/dt =  β S I / N - γ I
        dR/dt =  γ I

    Setting all derivatives to zero with S + I + R = N and S, I, R ≥ 0
    yields I* = 0 and a line of disease-free equilibria. The standard
    disease-free equilibrium (DFE) used here is:

        S* = N,  I* = 0,  R* = 0.

    We also return the critical susceptible level S_crit at which
    dI/dt = 0 given I > 0,

        S_crit = (γ / β) * N,

    and its fraction of the population S_crit / N = γ / β = 1 / R0.
    """
    # Disease-free equilibrium (all susceptible, no infection, no recovered)
    S_eq = float(N)
    I_eq = 0.0
    R_eq = 0.0

    # Herd-immunity / no-growth threshold for I(t)
    if beta > 0 and gamma > 0:
        S_crit = float(gamma * N / beta)
        S_crit_fraction = float(gamma / beta)  # == 1 / R0
    else:
        S_crit = np.nan
        S_crit_fraction = np.nan

    return S_eq, I_eq, R_eq, S_crit, S_crit_fraction

# ---------------------------------------------------------

def summarize_trajectory(t_array, S, I, R, beta, gamma,
                         contact_rate, infection_prob, I0):
    peak_I = float(I.max())
    time_to_peak = float(t_array[I.argmax()])
    total_infected = float(R[-1])      # final recovered ~ total infected
    attack_rate = total_infected / float(N)
    R0_eff = float(beta / gamma) if gamma > 0 else np.nan

    # NEW: analytic equilibrium & critical susceptible level
    S_eq, I_eq, R_eq, S_crit, S_crit_fraction = compute_equilibrium_points(
        beta, gamma, N
    )

    return {
        "I0": float(I0),
        "peak_I": peak_I,
        "peak_I_fraction": peak_I / float(N),
        "time_to_peak": time_to_peak,
        "total_infected": total_infected,
        "attack_rate": attack_rate,
        "beta": float(beta),
        "gamma": float(gamma),
        "contact_rate": float(contact_rate),
        "infection_prob": float(infection_prob),
        "R0_eff": R0_eff,
        # NEW fields
        "S_eq": S_eq,
        "I_eq": I_eq,
        "R_eq": R_eq,
        "S_crit": S_crit,
        "S_crit_fraction": S_crit_fraction,
    }

def save_time_series_csv(base_name, t_array, S, I, R,
                         scale_c, scale_p, scale_g,
                         beta, gamma, contact_rate, infection_prob, I0):
    """
    Save time-series data with the requested filename format:
    sir_<number_initially_infected>_initial_infection_<transmission_rate>_<infection_rate>_<recovery_rate>.csv
    """
    df = pd.DataFrame({
        "t": t_array.astype(float),
        "S": S.astype(float),
        "I": I.astype(float),
        "R": R.astype(float),
        "scale_contact": float(scale_c),
        "scale_infection_prob": float(scale_p),
        "scale_gamma": float(scale_g),
        "beta": float(beta),
        "gamma": float(gamma),
        "contact_rate": float(contact_rate),
        "infection_prob": float(infection_prob),
        "I0": float(I0),
        "N": float(N),
    })

    # Ensure ALL numeric columns are floats
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].astype(float)

    filename = (
        f"sir_{I0}_initial_infection_"
        f"contact_rate_{contact_rate:.4f}_infection_prob_{infection_prob:.4f}_beta_{beta:.4f}_gamma_{gamma:.4f}.csv"
    )
    path = os.path.join(DATA_DIR, filename)
    df.to_csv(path, index=False)
    return path

def make_sir_plot(base_name, t_array, S, I, R,
                  scale_c, scale_p, scale_g,
                  beta, gamma, contact_rate, infection_prob, I0):
    """
    Save timeseries PNG with name:
    sir_<I0>_initial_infection_<transmission>_<infection>_<recovery>_timeseries.png
    """
    with MATPLOTLIB_LOCK:
        plt.figure(figsize=(6, 4))
        plt.plot(t_array, S, label="Susceptible")
        plt.plot(t_array, I, label="Infected")
        plt.plot(t_array, R, label="Recovered")

        title = rf"SIR on Campus ($I_0$ = {I0}, N = {N})"
        subtitle = (
            f"Contact Scale = {scale_c:.2f}, p-scale = {scale_p:.2f}, "
            rf"$\gamma$ Scale = {scale_g:.2f}, "
            "\n"
            rf"$\beta$ = {beta:.4f}, $\gamma$ = {gamma:.4f}, "
            f"c = {contact_rate:.2f}, p = {infection_prob:.4f}"
        )
        plt.title(title + "\n" + subtitle)
        plt.xlabel("Time (days)")
        plt.ylabel("Number of individuals")
        plt.grid()
        plt.legend()
        plt.tight_layout()

        filename = (
            f"sir_{I0}_initial_infection_"
            f"contact_rate{contact_rate:.4f}_infection_prob_{infection_prob:.4f}_beta_{beta:.4f}_gamma_{gamma:.4f}_timeseries.png"
        )
        path = os.path.join(PLOT_DIR, filename)
        plt.savefig(path, dpi=400, bbox_inches="tight")
        plt.close()
    return path

def _factor_near_square(n: int):
    """Find (rows, cols) with rows * cols == n, close to sqrt(n)."""
    root = int(np.floor(np.sqrt(n)))
    for rows in range(root, 0, -1):
        if n % rows == 0:
            cols = n // rows
            return rows, cols
    return 1, n

def make_heatmap_animation(base_name, t_array, S_arr, I_arr, R_arr,
                           beta, gamma, contact_rate, infection_prob, I0):
    """
    Creates an animated grid where each colored square corresponds to one individual,
    and the grid has exactly N squares (no extra background cells).

    If MAKE_ANIMATIONS is False, returns None immediately.
    """
    if not MAKE_ANIMATIONS:
        return None

    n_rows, n_cols = _factor_near_square(N)
    cmap = ListedColormap(["#999999", "#E69F00", "#0072B2"])

    with MATPLOTLIB_LOCK:
        fig, ax = plt.subplots(figsize=(10, 10))
        initial_grid = np.zeros((n_rows, n_cols), dtype=int)
        im = ax.imshow(initial_grid, aspect="equal", cmap=cmap, vmin=0, vmax=2)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
        ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        ax.tick_params(which="minor", bottom=False, left=False,
                       labelbottom=False, labelleft=False)
        ax.set_axisbelow(False)

        legend_elements = [
            Patch(facecolor="#999999", label="Susceptible"),
            Patch(facecolor="#E69F00", label="Infected"),
            Patch(facecolor="#0072B2", label="Recovered"),
        ]
        ax.legend(handles=legend_elements, loc="upper right", framealpha=0.8)

        def update(frame):
            S_val = float(S_arr[frame])
            I_val = float(I_arr[frame])
            R_val = float(R_arr[frame])

            n_S = int(round(S_val))
            n_I = int(round(I_val))
            if n_S < 0:
                n_S = 0
            if n_I < 0:
                n_I = 0
            if n_S + n_I > N:
                excess = n_S + n_I - N
                reduce_I = min(excess, n_I)
                n_I -= reduce_I
                excess -= reduce_I
                if excess > 0:
                    n_S = max(0, n_S - excess)
            n_R = N - n_S - n_I
            if n_R < 0:
                n_R = 0

            statuses = np.zeros(N, dtype=int)
            statuses[n_S:n_S + n_I] = 1
            statuses[n_S + n_I:] = 2

            grid = statuses.reshape(n_rows, n_cols)
            im.set_data(grid)

            title = (
                f"t = {t_array[frame]:.2f} days | "
                f"S = {int(S_val)}, I = {int(I_val)}, R = {int(R_val)}\n"
                rf"$I_0$ = {I0}, $\beta$ = {beta:.4f}, $\gamma$ = {gamma:.4f}, "
                f"c = {contact_rate:.2f}, p = {infection_prob:.4f}"
            )
            ax.set_title(title, fontsize=9)
            return [im]

        ani = animation.FuncAnimation(
            fig, update, frames=len(t_array), blit=True, interval=100
        )

        filename = (
            f"sir_{I0}_initial_infection_"
            f"contact_rate_{contact_rate:.4f}_infection_prob_{infection_prob:.4f}_beta_{beta:.4f}_gamma_{gamma:.4f}_heatmap.gif"
        )
        gif_path = os.path.join(ANIM_DIR, filename)

        from matplotlib.animation import PillowWriter
        writer = PillowWriter(fps=8)
        ani.save(gif_path, writer=writer)

        plt.close(fig)

    return gif_path

# ---------------------------------------------------------
# NEW: β–γ peak infection heatmaps (absolute, per I0)
# ---------------------------------------------------------

def generate_peak_infection_heatmaps(summary_df: pd.DataFrame):
    """
    Generate static β–γ heatmaps showing peak infected (absolute count) for each I0.

    - x-axis: beta (transmission rate)
    - y-axis: gamma (recovery rate)
    - color axis (z): maximum number of infected individuals (peak_I)

    Each I0 gets its own PNG:
      sir_heatmap_peakI_I0_<I0>.png
    """
    print("\nGenerating β–γ peak infection heatmaps for each I0...")

    unique_I0_values = sorted(summary_df["I0"].unique())

    for I0_val in unique_I0_values:
        sub = summary_df[summary_df["I0"] == I0_val].copy()
        if sub.empty:
            continue

        betas = np.sort(sub["beta"].unique())
        gammas = np.sort(sub["gamma"].unique())

        # Matrix of peak infections (rows = gamma, cols = beta)
        Z = np.full((len(gammas), len(betas)), np.nan, dtype=float)

        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                mask = (sub["gamma"] == g) & (sub["beta"] == b)
                if mask.any():
                    Z[i, j] = float(sub.loc[mask, "peak_I"].max())

        with MATPLOTLIB_LOCK:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[betas.min(), betas.max(), gammas.min(), gammas.max()]
            )
            cbar = plt.colorbar(im)
            cbar.set_label("Peak infected (individuals)")

            plt.xlabel(r"$\beta$ (transmission rate)")
            plt.ylabel(r"$\gamma$ (recovery rate)")
            plt.title(rf"Peak Infections Heatmap ($I_0$ = {int(round(I0_val))})")
            plt.tight_layout()

            heatmap_filename = os.path.join(
                PLOT_DIR,
                f"sir_heatmap_peakI_I0_{int(round(I0_val))}.png"
            )
            plt.savefig(heatmap_filename, dpi=400, bbox_inches="tight")
            plt.close()

            print(f"  Saved heatmap for I0 = {int(round(I0_val))} to:")
            print(f"    {heatmap_filename}")

# ---------------------------------------------------------
# NEW: β–γ peak infection FRACTION heatmaps (per I0)
# ---------------------------------------------------------

def generate_peak_infection_heatmaps_fraction(summary_df: pd.DataFrame):
    """
    Generate static β–γ heatmaps showing peak infected as a FRACTION of the population
    for each I0.

    - x-axis: beta (transmission rate)
    - y-axis: gamma (recovery rate)
    - color axis (z): peak_I_fraction (peak infected / N)

    Each I0 gets its own PNG:
      sir_heatmap_peakI_fraction_I0_<I0>.png
    """
    print("\nGenerating β–γ peak infection FRACTION heatmaps for each I0...")

    unique_I0_values = sorted(summary_df["I0"].unique())

    for I0_val in unique_I0_values:
        sub = summary_df[summary_df["I0"] == I0_val].copy()
        if sub.empty:
            continue

        betas = np.sort(sub["beta"].unique())
        gammas = np.sort(sub["gamma"].unique())

        # Matrix of peak infection fractions (rows = gamma, cols = beta)
        Z = np.full((len(gammas), len(betas)), np.nan, dtype=float)

        for i, g in enumerate(gammas):
            for j, b in enumerate(betas):
                mask = (sub["gamma"] == g) & (sub["beta"] == b)
                if mask.any():
                    Z[i, j] = float(sub.loc[mask, "peak_I_fraction"].max())

        with MATPLOTLIB_LOCK:
            plt.figure(figsize=(6, 5))
            im = plt.imshow(
                Z,
                origin="lower",
                aspect="auto",
                extent=[betas.min(), betas.max(), gammas.min(), gammas.max()]
            )
            cbar = plt.colorbar(im)
            cbar.set_label("Peak infected (fraction of population)")

            plt.xlabel(r"$\beta$ (transmission rate)")
            plt.ylabel(r"$\gamma$ (recovery rate)")
            plt.title(rf"Peak Infection Fraction Heatmap ($I_0$ = {int(round(I0_val))})")
            plt.tight_layout()

            heatmap_filename = os.path.join(
                PLOT_DIR,
                f"sir_heatmap_peakI_fraction_I0_{int(round(I0_val))}.png"
            )
            plt.savefig(heatmap_filename, dpi=400, bbox_inches="tight")
            plt.close()

            print(f"  Saved fraction heatmap for I0 = {int(round(I0_val))} to:")
            print(f"    {heatmap_filename}")

# ---------------------------------------------------------
# 3. Worker for parallel execution (single scenario)
# ---------------------------------------------------------

def run_single_scenario(task):
    """
    Worker function for a single scenario.
    task = (scenario_id, I0, scale_c, scale_p, scale_g)
    Returns a summary dict, including per-scenario wall time.
    """
    scenario_id, I0, scale_c, scale_p, scale_g = task
    scenario_start = time.perf_counter()

    S0_base = N - I0 - R0_initial
    if S0_base < 0:
        raise ValueError(f"I0 = {I0} is too large for population N = {N}")

    contact_rate = contact_rate0 * scale_c
    infection_prob = infection_prob0 * scale_p
    gamma = gamma0 * scale_g
    beta = contact_rate * infection_prob

    S0 = S0_base

    S, I, R = run_sir_simulation(
        beta=beta,
        gamma=gamma,
        N=N,
        S0=S0,
        I0=I0,
        R0_initial=R0_initial,
        t_array=t
    )

    base_name = (
        f"I{I0}_c{fmt_scale(scale_c)}"
        f"_p{fmt_scale(scale_p)}_g{fmt_scale(scale_g)}"
    )

    csv_path = save_time_series_csv(
        base_name, t, S, I, R,
        scale_c, scale_p, scale_g,
        beta, gamma, contact_rate, infection_prob, I0
    )

    png_path = make_sir_plot(
        base_name, t, S, I, R,
        scale_c, scale_p, scale_g,
        beta, gamma, contact_rate, infection_prob, I0
    )

    gif_path = make_heatmap_animation(
        base_name, t, S, I, R,
        beta, gamma, contact_rate, infection_prob, I0
    )

    summary = summarize_trajectory(
        t, S, I, R,
        beta, gamma, contact_rate, infection_prob, I0
    )

    scenario_end = time.perf_counter()
    scenario_wall_time = scenario_end - scenario_start

    summary.update({
        "scenario_id": float(scenario_id),
        "scale_contact": float(scale_c),
        "scale_infection_prob": float(scale_p),
        "scale_gamma": float(scale_g),
        "csv_path": csv_path,
        "plot_path": png_path,
        "heatmap_path": gif_path if MAKE_ANIMATIONS else None,
        "scenario_wall_time": float(scenario_wall_time),
    })
    return summary

# ---------------------------------------------------------
# 4. Main (batches of 50, remainder as final smaller batch)
# ---------------------------------------------------------

def main():
    global_start = time.time()

    # ±10% in 5% steps
    scales = np.round(np.arange(0.90, 1.10 + 0.001, 0.05), 2)

    tasks = []
    scenario_id = 0
    for I0 in INITIAL_INFECTED_VALUES:
        for scale_c, scale_p, scale_g in itertools.product(scales, scales, scales):
            scenario_id += 1
            tasks.append((scenario_id, I0, scale_c, scale_p, scale_g))

    total = len(tasks)
    print(f"Total scenarios to simulate: {total}")
    print(f"GIF generation enabled? {MAKE_ANIMATIONS}")

    # Process in batches of 50. If total is not divisible by 50,
    # the final batch will contain the remaining (smaller) number of scenarios.
    BATCH_SIZE = 50

    summary_rows = []
    completed = 0
    batch_index = 0

    for batch_start in range(0, total, BATCH_SIZE):
        batch_index += 1
        batch_tasks = tasks[batch_start:batch_start + BATCH_SIZE]
        batch_count = len(batch_tasks)

        print(f"\n=== Starting batch {batch_index}: {batch_count} scenarios ===")

        # Fixed thread count based on your CPU: up to 24 threads per batch
        n_threads = min(24, batch_count)
        print(f"Using {n_threads} worker threads for batch {batch_index}...")

        with ThreadPoolExecutor(max_workers=n_threads) as executor:
            future_to_task = {
                executor.submit(run_single_scenario, task): task
                for task in batch_tasks
            }

            for future in as_completed(future_to_task):
                summary = future.result()
                summary_rows.append(summary)

                completed += 1
                if completed % 25 == 0 or completed == total:
                    print(f"  Completed {completed} / {total} scenarios...")

    summary_df = pd.DataFrame(summary_rows)

    numeric_cols = summary_df.select_dtypes(include=["number"]).columns
    summary_df[numeric_cols] = summary_df[numeric_cols].astype(float)

    summary_df.sort_values("scenario_id", inplace=True)

    summary_path = os.path.join(DATA_DIR, "sir_sweep_summary_all_scenarios.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved summary of all scenarios to:\n  {summary_path}")

    # Generate static β–γ heatmaps (absolute and fraction), separate from animation
    generate_peak_infection_heatmaps(summary_df)
    generate_peak_infection_heatmaps_fraction(summary_df)

    global_end = time.time()
    total_wall = global_end - global_start

    if "scenario_wall_time" in summary_df.columns:
        total_scenario_time = float(summary_df["scenario_wall_time"].sum())
        max_scenario_time = float(summary_df["scenario_wall_time"].max())
        avg_scenario_time = float(summary_df["scenario_wall_time"].mean())

        print("\n=== Timing Summary ===")
        print(f"Total wall-clock time           : {total_wall:.2f} seconds")
        print(f"Total scenario compute time     : {total_scenario_time:.2f} seconds")
        print(f"Average scenario compute time   : {avg_scenario_time:.2f} seconds")
        print(f"Max single-scenario wall time   : {max_scenario_time:.2f} seconds")

        if total_wall > 0:
            parallel_factor = total_scenario_time / total_wall
            print(f"Estimated parallel utilization  : {parallel_factor:.2f} core-seconds per second")
            print("(Higher numbers mean more effective parallelism.)")
    else:
        print(f"Total wall-clock time: {total_wall:.2f} seconds")

    print("Done.")

# ---------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------

if __name__ == "__main__":
    main()
