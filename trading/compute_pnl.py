import numpy as np
import pandas as pd
from typing import Tuple, Optional

def compute_frc_signal(frc_dart_col:np.array, tol:float):
    """
    Forecast signal: long/short if forecasted hour postitive/negative.
    """
    signal = np.where(
        np.abs(frc_dart_col) < tol, 0, 
        np.where(frc_dart_col > 0, 1, -1))
    return signal

def compute_naive_signal(actual_dart_col:np.array, tol:float):
    """
    Naive signal: long/short if same hour yesterday positive/negative.
    """
    shifted = np.zeros_like(actual_dart_col)
    shifted[1:] = actual_dart_col[:-1]
    
    signal = np.where(
        np.abs(shifted) < tol, 0, 
        np.where(shifted > 0, 1, -1))

    return signal

def timestep_pnl(actual_dart_col: np.array, signal_col: np.array):
    """Calculate PnL for a single timestep."""
    return actual_dart_col * signal_col

def trade_pnl(strategy_name:str,
              start_end_date_test:Tuple[str, str], 
              start_end_hour_trade:Tuple[int, int],
              # dummy arg for now
              init_budget:int, 
              hourly_volume:float, 
              actual_dart_col_name:str,
              tol:float,
              df:pd.DataFrame,
              # this being None performs the
              # naive strategy
              forecast_dart_col_name:Optional[str]=None,
              date_col='DATE'):
        
        start_date = pd.to_datetime(start_end_date_test[0])
        end_date = pd.to_datetime(start_end_date_test[1])
            
        mask = (df['DATE'] >= start_date) & (df['DATE'] <= end_date)
        df = df.loc[mask].reset_index(drop=True)

        actual_dart_col = np.asarray(df[actual_dart_col_name])
        forecast_dart_col = None
        if forecast_dart_col_name is not None:
            forecast_dart_col = np.asarray(df[forecast_dart_col_name])
        
        signal = None 
        if forecast_dart_col is None:
            signal = compute_naive_signal(actual_dart_col, tol)
        else:
            signal = compute_frc_signal(forecast_dart_col, tol)
            
        hours = np.arange(len(signal)) % 24
        hour_mask = ((hours >= start_end_hour_trade[0]) & (hours <= start_end_hour_trade[1])).astype(int)
        
        pnl = hourly_volume * actual_dart_col * signal * hour_mask
        cumulative_pnl = np.cumsum(pnl)
        
        return {
            'name': strategy_name,
            'date_idx': df[date_col],
            'PnL': pnl,
            'cumulative_PnL': cumulative_pnl,
            'signal': signal
        }