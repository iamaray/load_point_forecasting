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

def compute_metrics(strategy_outcomes:List[dict])->List[dict]:
    
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