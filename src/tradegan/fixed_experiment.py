"""Runnable TradeGAN experiment orchestration using extracted modules."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from .data.splits import split_train_val_test, split_train_val_testraw
from .models.gan import Generator, Discriminator
from .lstm_model import LSTMForecaster
from .objectives.losses import GradientCheck, GradientCheckLSTM
from .evaluation.gan import Evaluation2
from .evaluation.lstm import Evaluation2LSTM
from .training.gan import *
from .training.lstm import *

GAN_OBJECTIVES={"PnL":TrainLoopMainPnLnv,"PnL MSE":TrainLoopMainPnLMSEnv,"PnL MSE STD":TrainLoopMainPnLMSESTDnv,"PnL MSE SR":TrainLoopMainPnLMSESRnv,"PnL SR":TrainLoopMainPnLSRnv,"PnL STD":TrainLoopMainPnLSTDnv,"SR":TrainLoopMainSRnv,"SR MSE":TrainLoopMainSRMSEnv,"MSE":TrainLoopMainMSEnv}
LSTM_OBJECTIVES={"PnL":TrainLoopnLSTMPnL,"PnL STD":TrainLoopnLSTMPnLSTD,"PnL SR":TrainLoopnLSTMPnLSR,"STD":TrainLoopnLSTMSTD,"SR":TrainLoopnLSTMSR,"MSE":TrainLoopnLSTM}


def _device(use_gpu=True):
    return torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")


def _safe_stats(x):
    return x.mean(), x.std().clamp_min(1e-8)


def _split_tcs(ticker,data_dir,metadata,tr=.8,vl=.1):
    if ticker.startswith("X"):
        return split_train_val_testraw(ticker,str(data_dir)+"/",tr=tr,vl=vl,h=1,l=10,pred=1,plotcheck=False)
    return split_train_val_test(ticker,str(data_dir)+"/",str(metadata),tr=tr,vl=vl,h=1,l=10,pred=1,plotcheck=False)


def _new_gan(train,device,z_dim,hid_g,hid_d,lookback,pred,lr_g,lr_d):
    mean,std=_safe_stats(train)
    g=Generator(z_dim,lookback,hid_g,pred,mean,std).to(device)
    d=Discriminator(lookback+pred,hid_d,mean,std).to(device)
    return g,d,torch.optim.RMSprop(g.parameters(),lr=lr_g),torch.optim.RMSprop(d.parameters(),lr=lr_d)


def _save_curve(path,values,title):
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True,exist_ok=True)
    plt.figure(figsize=(10,5)); plt.plot(np.cumsum(values)); plt.title(title); plt.xlabel("test period"); plt.ylabel("cumulative PnL (bps)"); plt.tight_layout(); plt.savefig(path); plt.close()


def run_gan_experiment(ticker,root,gan_epochs=100,gradient_epochs=100,batch_size=100,lr_g=1e-4,lr_d=1e-4,lookback=10,pred=1,z_dim=8,hid_g=8,hid_d=8,tanh_coeff=100,tr=.8,vl=.1,diter=1,use_gpu=True):
    root=Path(root); metrics=root/"results/metrics"; ckpt=root/"results/checkpoints"; figs=root/"results/figures"
    for p in (metrics,ckpt,figs): p.mkdir(parents=True,exist_ok=True)
    a,b,c,_=_split_tcs(ticker,root/"data",root/"stocks-etfs-list.csv",tr,vl); device=_device(use_gpu)
    train=torch.from_numpy(a).float().to(device); val=torch.from_numpy(b).float().to(device); test=torch.from_numpy(c).float().to(device); criterion=nn.BCELoss()
    rows=[]
    for name,trainer in {**GAN_OBJECTIVES,"ForGAN":TrainLoopForGAN}.items():
        print(f"[{ticker}] GAN objective: {name}"); g,d,go,do=_new_gan(train,device,z_dim,hid_g,hid_d,lookback,pred,lr_g,lr_d)
        if name!="ForGAN":
            g,d,go,do,alpha,beta,gamma,delta=GradientCheck(ticker,g,d,go,do,criterion,gradient_epochs,train,batch_size,hid_d,hid_g,z_dim,lr_d,lr_g,1,lookback,pred,diter,tanh_coeff,device,False)
            g,d,go,do=trainer(g,d,go,do,criterion,alpha,beta,gamma,delta,gan_epochs,20,train,val,batch_size,hid_d,hid_g,z_dim,lr_d,lr_g,1,lookback,pred,diter,tanh_coeff,device,False)
        else:
            g,d,go,do=trainer(g,d,go,do,criterion,alpha=0,beta=0,gamma=0,delta=0,n_epochs=gan_epochs,checkpoint_epoch=0,train_data=train,validation_data=val,batch_size=batch_size,hid_d=hid_d,hid_g=hid_g,z_dim=z_dim,lr_d=lr_d,lr_g=lr_g,h=1,l=lookback,pred=pred,diter=diter,tanh_coeff=tanh_coeff,device=device,plot=False)
        torch.save({"model_state_dict":g.state_dict(),"optimizer_state_dict":go.state_dict(),"objective":name,"ticker":ticker},ckpt/f"{ticker}-GAN-{name.replace(' ','_')}-generator.pt")
        df,pnl,*_=Evaluation2(ticker,2,g,test,val,1,lookback,pred,hid_d,hid_g,z_dim,lr_g,lr_d,gan_epochs,name,0,device,str(figs)+"/",name,False)
        row=df.iloc[0].to_dict(); row["objective"]=name; rows.append(row); pnl=np.asarray(pnl); _save_curve(figs/f"{ticker}-GAN-{name.replace(' ','_')}-cumulative-pnl.png",pnl,f"Cumulative PnL — {ticker} — {name}")
        pd.DataFrame(pnl).to_csv(metrics/f"{ticker}-GAN-{name.replace(' ','_')}-pnl.csv",index=False)
    out=pd.DataFrame(rows); out.to_csv(metrics/f"{ticker}-GAN-results.csv",index=False); return out


def run_lstm_experiment(ticker,root,lstm_epochs=500,gradient_epochs=100,batch_size=100,lr=1e-4,lookback=10,pred=1,hid_g=8,tanh_coeff=100,tr=.8,vl=.1,use_gpu=True):
    root=Path(root); metrics=root/"results/metrics"; ckpt=root/"results/checkpoints"; figs=root/"results/figures"
    for p in (metrics,ckpt,figs): p.mkdir(parents=True,exist_ok=True)
    a,b,c,_=_split_tcs(ticker,root/"data",root/"stocks-etfs-list.csv",tr,vl); device=_device(use_gpu)
    train=torch.from_numpy(a).float().to(device); val=torch.from_numpy(b).float().to(device); test=torch.from_numpy(c).float().to(device); mean,std=_safe_stats(train); rows=[]
    for name,trainer in LSTM_OBJECTIVES.items():
        print(f"[{ticker}] LSTM objective: {name}"); g=LSTMForecaster(0,lookback,hid_g,pred,mean,std).to(device); opt=torch.optim.RMSprop(g.parameters(),lr=lr)
        g,opt,alpha,beta,gamma,delta=GradientCheckLSTM(ticker,g,opt,gradient_epochs,train,batch_size,hid_g,hid_g,0,0,lr,1,lookback,pred,1,tanh_coeff,device,False)
        g,opt=trainer(g,opt,False,alpha,beta,gamma,delta,lstm_epochs,20,train,val,batch_size,hid_g,hid_g,0,lr,lr,1,lookback,pred,1,tanh_coeff,device,False)
        torch.save({"model_state_dict":g.state_dict(),"optimizer_state_dict":opt.state_dict(),"objective":name,"ticker":ticker},ckpt/f"{ticker}-LSTM-{name.replace(' ','_')}-generator.pt")
        df,pnl,*_=Evaluation2LSTM(ticker,2,g,test,val,1,lookback,pred,hid_g,hid_g,0,lr,lr,lstm_epochs,name,0,device,str(figs)+"/",name,False)
        row=df.iloc[0].to_dict(); row["objective"]=name; rows.append(row); pnl=np.asarray(pnl); _save_curve(figs/f"{ticker}-LSTM-{name.replace(' ','_')}-cumulative-pnl.png",pnl,f"Cumulative PnL — {ticker} — LSTM {name}")
        pd.DataFrame(pnl).to_csv(metrics/f"{ticker}-LSTM-{name.replace(' ','_')}-pnl.csv",index=False)
    out=pd.DataFrame(rows); out.to_csv(metrics/f"{ticker}-LSTM-results.csv",index=False); return out
