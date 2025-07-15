import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict


def profit_factor(pnl:np.array):
    pos = pnl * (pnl > 0)
    neg = pnl * (pnl < 0)
    
    profit_factor = np.sum(pos) / np.abs(np.sum(neg))
    return profit_factor

def drawdown(cumulative_pnl:np.array):
    cummax_pnl = np.maximum.accumulate(cumulative_pnl)
    return (cumulative_pnl - cummax_pnl) / (cummax_pnl + 1e-8)

def sharpe_ratio(mu_pnl:float, sig_pnl:float):
    sharpe = mu_pnl / (sig_pnl + 1e-8)
    return sharpe * np.sqrt(24 * 365)

def sortino_ratio(pnl:np.array, mu_pnl:float, mar:float=0.0):
    bad_dev = np.min(0, pnl - mar)
    downside_std = np.sqrt(np.mean(bad_dev) ** 2)
    sortino = ((mu_pnl - mar) / downside_std)
    
    return sortino * np.sqrt(24 * 365)

def active_info_ratio(pnl1:np.array, pnl2:np.array):
    pass

def compute_metrics(strategy_outcomes:List[dict]):
    
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

def plot_hourly_pnl(strategy_outcomes:List[dict]):
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

def plot_cumulative_pnl(strategy_outcomes:List[dict], out_filename:str="cumulative_pnl_comparison.png"):
    output_dir = "results/plots/trading"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, out_filename)

    plt.figure(figsize=(12, 6))
    for data in strategy_outcomes:
        plt.plot(data['date_idx'], data['cumulative_PnL'], label=data.get('name', 'Strategy'))
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL')
    plt.title('Cumulative PnL Comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    

def sharpe_barchart(strategy_outcomes:List[dict], out_filename:str="cumulative_pnl_comparison.png"):
    pass

def sortino_barchart(strategy_outcomes:List[dict], out_filename:str="cumulative_pnl_comparison.png"):
    pass

def profit_factor_barchart(strategy_outcomes:List[dict], out_filename:str="cumulative_pnl_comparison.png"):
    pass

