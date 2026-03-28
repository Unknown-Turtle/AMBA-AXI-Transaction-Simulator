"""
=============================================================================
Visualisations
=============================================================================
Reads the CSV outputs from the triple-master (no QoS) and QoS simulations
and generates comparison charts.

Plots generated:
  1. Per-master average latency: No-QoS vs QoS (grouped bar chart)
  2. Transaction latency over time, per master (No-QoS vs QoS)
  3. Queue depth over time (No-QoS vs QoS)
  4. Traffic injection profile showing each master's activity phases

Usage: python visualise.py
Output: saves PNGs to plots/ directory and displays them
=============================================================================
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Style setup ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f0f0f',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#333355',
    'axes.labelcolor': '#cccccc',
    'text.color': '#cccccc',
    'xtick.color': '#999999',
    'ytick.color': '#999999',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.6,
    'font.family': 'monospace',
    'font.size': 10,
})


COLORS = {
    'CPU': '#4fc3f7',   
    'GPU': '#ff7043', 
    'NPU': '#66bb6a',  
}

PLOT_DIR = 'plots'
os.makedirs(PLOT_DIR, exist_ok=True)


# ── Load data ────────────────────────────────────────────────────────────

def load_csv(path):
    if not os.path.exists(path):
        print(f"  [!] {path} not found. Run the simulation first.")
        return None
    return pd.read_csv(path)


df_noqos = load_csv('triple_master_logs.csv')
df_qos   = load_csv('qos_logs.csv')

if df_noqos is None or df_qos is None:
    print("Missing CSV files. Run amba_axi_triple_master.py and amba_axi_qos.py first.")
    exit(1)


# ===========================================================================
# PLOT 1: Average Latency Comparison (Grouped Bar Chart)
# ===========================================================================

def plot_latency_comparison():
    fig, ax = plt.subplots(figsize=(10, 6))

    masters = ['CPU', 'NPU', 'GPU']
    noqos_lats = [df_noqos[df_noqos['Source'] == m]['Latency_Cycles'].mean() for m in masters]
    qos_lats   = [df_qos[df_qos['Source'] == m]['Latency_Cycles'].mean() for m in masters]

    x = np.arange(len(masters))
    width = 0.35

    bars1 = ax.bar(x - width/2, noqos_lats, width, label='No QoS (FIFO)',
                   color=['#4fc3f7aa', '#66bb6aaa', '#ff7043aa'],
                   edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, qos_lats, width, label='With QoS (GUARANTEED)',
                   color=['#4fc3f7', '#66bb6a', '#ff7043'],
                   edgecolor='white', linewidth=0.5)

    # value labels on bars
    for bar, val in zip(bars1, noqos_lats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='#aaaaaa')
    for bar, val in zip(bars2, qos_lats):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val:.0f}', ha='center', va='bottom', fontsize=9, color='white')

    ax.set_xlabel('Master')
    ax.set_ylabel('Average Latency (cycles)')
    ax.set_title('QoS Impact: Average Transaction Latency per Master',
                 fontsize=13, fontweight='bold', color='white')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{m}\n({["Cortex-A","Ethos-N","Mali"][i]})'
                        for i, m in enumerate(masters)])
    ax.legend(loc='upper left', framealpha=0.3)
    ax.grid(axis='y', linestyle='--')
    ax.set_axisbelow(True)

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '1_latency_comparison.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# PLOT 2: Latency Over Time (Rolling Average, No-QoS vs QoS side by side)
# ===========================================================================

def plot_latency_over_time():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

    window = 200  # rolling window for smoothing

    for ax, df, title in [
        (ax1, df_noqos, 'No QoS (FIFO)'),
        (ax2, df_qos,   'With QoS (GUARANTEED)'),
    ]:
        for master in ['CPU', 'NPU', 'GPU']:
            subset = df[df['Source'] == master].copy()
            subset = subset.sort_values('End_Cycle')
            # rolling average of latency over completion order
            rolling = subset['Latency_Cycles'].rolling(window=window, min_periods=50).mean()
            ax.plot(subset['End_Cycle'].values, rolling.values,
                    color=COLORS[master], label=master, linewidth=1.2, alpha=0.9)

        ax.set_xlabel('Clock Cycle')
        ax.set_title(title, fontsize=12, fontweight='bold', color='white')
        ax.legend(framealpha=0.3)
        ax.grid(True, linestyle='--')
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    ax1.set_ylabel('Latency (cycles, rolling avg)')

    fig.suptitle('Transaction Latency Over Time — Per Master',
                 fontsize=14, fontweight='bold', color='white', y=1.02)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '2_latency_over_time.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor(), bbox_inches='tight')
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# PLOT 3: Queue Depth Over Time
# ===========================================================================

def plot_queue_depth():
    fig, ax = plt.subplots(figsize=(12, 5))

    # Reconstruct queue depth from transaction data
    # Approximate by counting: txns created up to cycle - txns completed up to cycle
    max_cycle = 10_000

    for df, label, color, ls in [
        (df_noqos, 'No QoS (FIFO)', '#e06060', '-'),
        (df_qos,   'QoS (GUARANTEED)', '#60c0e0', '--'),
    ]:
        # bin transactions by creation and completion cycle
        cycles = np.arange(max_cycle)
        created_cum = np.zeros(max_cycle)
        completed_cum = np.zeros(max_cycle)

        for _, row in df.iterrows():
            sc = int(row['Start_Cycle'])
            ec = int(row['End_Cycle'])
            if sc < max_cycle:
                created_cum[sc] += 1
            if ec < max_cycle:
                completed_cum[ec] += 1

        created_cum = np.cumsum(created_cum)
        completed_cum = np.cumsum(completed_cum)
        queue_depth = created_cum - completed_cum

        # downsample for plotting performance
        step = 10
        ax.plot(cycles[::step], queue_depth[::step],
                color=color, label=label, linewidth=1.3, linestyle=ls, alpha=0.9)

    ax.set_xlabel('Clock Cycle')
    ax.set_ylabel('Queue Depth (transactions)')
    ax.set_title('Bus Queue Depth Over Time',
                 fontsize=13, fontweight='bold', color='white')
    ax.legend(framealpha=0.3)
    ax.grid(True, linestyle='--')
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k'))

    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '3_queue_depth.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# PLOT 4: Traffic Injection Profile (first 1000 cycles zoomed in)
# ===========================================================================

def plot_traffic_profile():
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

    zoom = 1000  # show first 1000 cycles for clarity
    df = df_noqos  # traffic pattern is the same for both sims

    for ax, master in zip(axes, ['CPU', 'NPU', 'GPU']):
        subset = df[(df['Source'] == master) & (df['Start_Cycle'] < zoom)]
        # count transactions created per cycle
        counts = subset.groupby('Start_Cycle').size()
        all_cycles = np.arange(zoom)
        traffic = np.zeros(zoom)
        for c, n in counts.items():
            if c < zoom:
                traffic[int(c)] = n

        ax.bar(all_cycles, traffic, width=1.0, color=COLORS[master], alpha=0.7,
               edgecolor='none')
        ax.set_ylabel(f'{master}\ntxns/cycle', fontsize=10)
        ax.set_ylim(0, max(traffic.max() * 1.2, 1))
        ax.grid(True, linestyle='--', axis='y')
        ax.set_axisbelow(True)

        # annotate pattern
        if master == 'CPU':
            ax.text(zoom * 0.97, traffic.max() * 0.85, 'Poisson bursts (always on)',
                    ha='right', fontsize=9, color=COLORS[master], fontstyle='italic')
        elif master == 'GPU':
            ax.text(zoom * 0.97, traffic.max() * 0.85, 'Sustained active / idle gaps',
                    ha='right', fontsize=9, color=COLORS[master], fontstyle='italic')
        elif master == 'NPU':
            ax.text(zoom * 0.97, traffic.max() * 0.85, 'Load → Compute (silent) → Store',
                    ha='right', fontsize=9, color=COLORS[master], fontstyle='italic')

    axes[-1].set_xlabel('Clock Cycle')
    fig.suptitle('Traffic Injection Profile (first 1,000 cycles)',
                 fontsize=13, fontweight='bold', color='white')
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, '4_traffic_profile.png')
    fig.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    print(f"  Saved: {path}")
    plt.close(fig)


# ===========================================================================
# RUN ALL
# ===========================================================================

if __name__ == '__main__':
    print("\n  Generating visualisations...\n")
    plot_latency_comparison()
    plot_latency_over_time()
    plot_queue_depth()
    plot_traffic_profile()
    print(f"\n  All plots saved to {PLOT_DIR}/\n")
