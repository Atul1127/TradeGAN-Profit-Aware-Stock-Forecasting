# Methodology

TradeGAN reproduces the Fin-GAN approach with recurrent generator/discriminator components and economics-aware objectives. The generator can combine adversarial BCE with MSE, PnL, Sharpe ratio, and volatility terms.

The forecasting target is half-day excess log-return relative to the benchmark series. The historical implementation uses a differentiable tanh surrogate for trading-sign objectives.

The original research method is attributed to Vuletić and Cont (2023); this repository is a reproduction/application.
