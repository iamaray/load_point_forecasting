import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict


def profit_factor(pnl: np.array):
    pos = pnl * (pnl > 0)
    neg = pnl * (pnl < 0)

    profit_factor = np.sum(pos) / np.abs(np.sum(neg))
    return profit_factor


def drawdown(cumulative_pnl: np.array):
    cummax_pnl = np.maximum.accumulate(cumulative_pnl)
    return (cumulative_pnl - cummax_pnl) / (cummax_pnl + 1e-8)


def sharpe_ratio(mu_pnl: float, sig_pnl: float):
    sharpe = mu_pnl / (sig_pnl + 1e-8)
    return sharpe * np.sqrt(24 * 365)


def sortino_ratio(pnl: np.array, mu_pnl: float, mar: float = 0.0):
    bad_dev = np.min(0, pnl - mar)
    downside_std = np.sqrt(np.mean(bad_dev) ** 2)
    sortino = ((mu_pnl - mar) / downside_std)

    return sortino * np.sqrt(24 * 365)


def active_info_ratio(pnl1: np.array, pnl2: np.array):
    pass


def compute_metrics(strategy_outcomes: List[dict]):

    for data in strategy_outcomes:
        pnl = data['PnL']
        cum_pnl = data['cumulative_PnL']

        mu_pnl = np.mean(pnl)
        sig_pnl = np.std(pnl)

        data['mu_PnL'] = mu_pnl
        data['sig_PnL'] = sig_pnl
        data['hit_rate'] = np.mean(pnl > 0)
        data['miss_rate'] = np.mean(pnl < 0)

        dd = drawdown(pnl)
        data['drawdown'] = dd
        data['mdd'] = np.min(dd)

        data['profit_factor'] = profit_factor(pnl)
        data['sharpe'] = sharpe_ratio(mu_pnl, sig_pnl)
        data['sortino'] = sortino_ratio(pnl, mu_pnl)


def plot_hourly_pnl(strategy_outcomes: List[dict]):
    output_dir = "results/plots/trading"
    os.makedirs(output_dir, exist_ok=True)
    for data in strategy_outcomes:
        name = data.get('name', 'Strategy')
        date_idx = data['date_idx']
        pnl = data['PnL']
        plt.figure(figsize=(12, 6))
        plt.plot(date_idx, pnl, label=f"{name} Hourly PnL")
        plt.xlabel('Date')
        plt.ylabel('PnL')
        plt.title(f'Hourly PnL for {name}')
        plt.legend()
        plt.tight_layout()
        out_filename = f"{name}_hourly_pnl.png".replace(" ", "_")
        output_path = os.path.join(output_dir, out_filename)
        plt.savefig(output_path)
        plt.close()


def plot_cumulative_pnl(strategy_outcomes: List[dict], out_filename: str = "cumulative_pnl_comparison.png"):
    output_dir = "results/plots/trading"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_filename)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for data in strategy_outcomes:
        ax1.plot(data['date_idx'], data['cumulative_PnL'],
                 label=data.get('name', 'Strategy'))
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative PnL')
    ax1.set_title('Cumulative PnL Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for data in strategy_outcomes:
        ax2.plot(data['date_idx'], data['drawdown'],
                 label=data.get('name', 'Strategy'))
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown')
    ax2.set_title('Drawdown Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def sharpe_barchart(strategy_outcomes: List[dict], out_filename: str = "sharpe_ratio_comparison.png"):
    output_dir = "results/plots/trading"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_filename)

    strategy_names = [
        data.get('name', f'Strategy_{i}') for i, data in enumerate(strategy_outcomes)]
    sharpe_ratios = [data['sharpe'] for data in strategy_outcomes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategy_names, sharpe_ratios, alpha=0.7)

    for bar, sharpe in zip(bars, sharpe_ratios):
        if sharpe >= 0:
            bar.set_color('green')
        else:
            bar.set_color('red')

    plt.xlabel('Strategy')
    plt.ylabel('Sharpe Ratio')
    plt.title('Sharpe Ratio Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, sharpe in zip(bars, sharpe_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                 f'{sharpe:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def sortino_barchart(strategy_outcomes: List[dict], out_filename: str = "sortino_ratio_comparison.png"):
    output_dir = "results/plots/trading"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_filename)

    strategy_names = [
        data.get('name', f'Strategy_{i}') for i, data in enumerate(strategy_outcomes)]
    sortino_ratios = [data['sortino'] for data in strategy_outcomes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategy_names, sortino_ratios, alpha=0.7)

    for bar, sortino in zip(bars, sortino_ratios):
        if sortino >= 0:
            bar.set_color('blue')
        else:
            bar.set_color('red')

    plt.xlabel('Strategy')
    plt.ylabel('Sortino Ratio')
    plt.title('Sortino Ratio Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    for bar, sortino in zip(bars, sortino_ratios):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height >= 0 else -0.05),
                 f'{sortino:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def profit_factor_barchart(strategy_outcomes: List[dict], out_filename: str = "profit_factor_comparison.png"):
    output_dir = "results/plots/trading"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_filename)

    strategy_names = [
        data.get('name', f'Strategy_{i}') for i, data in enumerate(strategy_outcomes)]
    profit_factors = [data['profit_factor'] for data in strategy_outcomes]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(strategy_names, profit_factors, alpha=0.7)

    for bar, pf in zip(bars, profit_factors):
        if pf >= 1:
            bar.set_color('orange')
        else:
            bar.set_color('red')

    plt.xlabel('Strategy')
    plt.ylabel('Profit Factor')
    plt.title('Profit Factor Comparison')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')

    plt.axhline(y=1, color='black', linestyle='--',
                alpha=0.5, label='Break-even')
    plt.legend()

    for bar, pf in zip(bars, profit_factors):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{pf:.2f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
