import numpy as np
import pandas as pd
import json
import os

from compare_strats import *
from compute_pnl import *


def compare(cfg_path: str):
    original_dir = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)

    with open(cfg_path, 'r') as f:
        config = json.load(f)

    params = config.get('params', {})
    forecasts = config.get('forecasts', {})

    data_lst = []

    for (k, v) in forecasts.items():
        df = pd.read_csv(v, index_col=0)
        df.reset_index(inplace=True)
        df.rename(columns={df.columns[0]: "DATE"}, inplace=True)
        df['DATE'] = pd.to_datetime(df['DATE'])

        data = trade_pnl(
            strategy_name=k,
            start_end_date_test=(params['test_start'], params['test_end']),
            start_end_hour_trade=(
                params['trade_start_hr'], params['trade_end_hr']),
            init_budget=params['init_budget'],
            hourly_volume=params['hourly_volume'],
            actual_dart_col_name=params['actual_dart_col_name'],
            tol=params['tol'],
            df=df,
            forecast_dart_col_name=params.get('forecast_dart_col_name', None),
            date_col=params.get('date_col_name', 'DATE')
        )

        data_lst.append(data)

    naive = trade_pnl(
        strategy_name='NAIVE',
        start_end_date_test=(params['test_start'], params['test_end']),
        start_end_hour_trade=(
            params['trade_start_hr'], params['trade_end_hr']),
        init_budget=params['init_budget'],
        hourly_volume=params['hourly_volume'],
        actual_dart_col_name=params['actual_dart_col_name'],
        tol=params['tol'],
        df=df,
        forecast_dart_col_name=None,
        date_col=params.get('date_col_name', 'DATE')
    )

    data_lst.append(naive)

    compute_metrics(data_lst)
    plot_hourly_pnl(data_lst)
    plot_cumulative_pnl(data_lst)
    sharpe_barchart(data_lst)
    sortino_barchart(data_lst)
    profit_factor_barchart(data_lst)
    max_drawdown_barchart(data_lst)

    os.chdir(original_dir)


if __name__ == "__main__":
    compare('cfgs/trading.json')
