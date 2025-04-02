# Point-forecasting models for electricity price and load forecasting

This codebase implements a collection of point forecasting DL/ML models for our research in power market demand and price prediction.

# Implemented Models

- **FFNN**: A standard feed-forward neural network.
- **LSTM**: A standard LSTM model.
- **Transformer**: A standard transformer model.
- **Informer**: An efficient transformer for time series forecasting. Zhou, Haoyi, et al. "Informer: Beyond efficient transformer for long sequence time-series forecasting." Proceedings of the AAAI conference on artificial intelligence. Vol. 35. No. 12. 2021.
- **FGN**: From the paper https://doi.org/10.48550/arXiv.2311.06190.
- **LSTM-Attention-LSTM**: From the paper X. Wen and W. Li, "Time Series Prediction Based on LSTM-Attention-LSTM Model," in IEEE Access, vol. 11, pp. 48322-48331, 2023, doi: 10.1109/ACCESS.2023.3276628.
- **TimeXer**: From the paper https://doi.org/10.48550/arXiv.2402.19072.
- **CATS**: Learned auxiliary time series; can be fit with any model. See the paper https://doi.org/10.48550/arXiv.2403.01673.

<!-- - **Autoregressive LSTM**: The same LSTM model but trained in an autoregressive fashion; that is, it takes its own previous outputs as input. -->
