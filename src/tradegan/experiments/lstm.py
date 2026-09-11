"""LSTM experiment entry points."""
from pathlib import Path
from tradegan.fixed_experiment import run_lstm_experiment


def LSTM_combos(ticker, loc, modelsloc, plotsloc, dataloc, etflistloc, vl_later=True,
                lrg=0.0001, lrd=0.0001, n_epochs=500, ngrad=100, h=1, l=10, pred=1,
                ngpu=1, tanh_coeff=100, tr=0.8, vl=0.1, z_dim=32, hid_d=64,
                hid_g=8, checkpoint_epoch=20, batch_size=100, diter=1, plot=False, freq=2):
    del loc, modelsloc, plotsloc, etflistloc, vl_later, lrd, h, ngpu, z_dim, hid_d, checkpoint_epoch, diter, plot, freq
    return run_lstm_experiment(ticker, Path(dataloc).resolve().parent,
                               lstm_epochs=n_epochs, gradient_epochs=ngrad, batch_size=batch_size,
                               lr=lrg, lookback=l, pred=pred, hid_g=hid_g,
                               tanh_coeff=tanh_coeff, tr=tr, vl=vl, use_gpu=True)

__all__=["LSTM_combos"]
