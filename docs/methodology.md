# Methodology

TradeGAN reproduces the Fin-GAN approach with recurrent generator/discriminator components and economics-aware objectives. The generator can combine adversarial BCE with MSE, PnL, Sharpe ratio, and volatility terms.

The forecasting target is half-day excess log-return relative to the benchmark series. The historical implementation uses a differentiable tanh surrogate for trading-sign objectives.

Loss-term weights come from gradient calibration: before training, each term's gradient size is measured on a copy of the model, and each weight is the median ratio of the base term's gradient size to that term's. Calibration does not update the models that are later trained.

Models train on returns clipped to ±15% per leg. Validation and test strategies are scored on the raw returns. Daily PnL is the overnight half-day PnL plus the following intraday half-day PnL. Every result row reports constant-position baselines and timing-only PnL, so that direction bias can be told apart from forecasting skill.

The original research method is attributed to Vuletić and Cont (2023); this repository is a reproduction/application.
