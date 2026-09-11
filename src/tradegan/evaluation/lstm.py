"""LSTM evaluation without legacy dependencies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch


def Evaluation2LSTM(ticker,freq,gen,test_data,val_data,h,l,pred,hid_d,hid_g,z_dim,lrg,lrd,n_epochs,losstype,sr_val,device,plotsloc,f_name,plot=False):
    del freq,hid_d,z_dim,lrg,lrd,n_epochs,sr_val,plotsloc,f_name,plot
    def run(data):
        n=data.shape[0]; cond=data[:,:l].unsqueeze(0).to(device=device,dtype=torch.float)
        h0=torch.zeros((1,n,hid_g),device=device); c0=torch.zeros_like(h0)
        gen.eval()
        with torch.no_grad(): out=gen(cond,h0,c0).squeeze(0).squeeze(-1)
        real=data[:,-1].to(device=device,dtype=torch.float); pnl=10000*torch.sign(out)*real
        weekly=pnl[:2*(len(pnl)//2)].reshape(-1,2).sum(1)
        return out,real,weekly,pnl
    tm,tr,tw,tp=run(test_data); vm,vr,vw,vp=run(val_data)
    corr=np.corrcoef(tm.cpu().numpy(),tr.cpu().numpy())[0,1]; corrv=np.corrcoef(vm.cpu().numpy(),vr.cpu().numpy())[0,1]
    dt={'lrd':lrd,'lrg':lrg,'type':losstype,'epochs':n_epochs,'ticker':ticker,
        'RMSE':torch.sqrt(torch.mean((tm-tr)**2)).item(),'MAE':torch.mean(torch.abs(tm-tr)).item(),
        'PnL_m test':tw.mean().item(),'SR_m scaled test':(tw.mean()/tw.std().clamp_min(torch.finfo(tw.dtype).eps)*torch.sqrt(torch.tensor(252.,device=device))).item(),
        'RMSE val':torch.sqrt(torch.mean((vm-vr)**2)).item(),'MAE val':torch.mean(torch.abs(vm-vr)).item(),
        'PnL_m val':vw.mean().item(),'SR_m scaled val':(vw.mean()/vw.std().clamp_min(torch.finfo(vw.dtype).eps)*torch.sqrt(torch.tensor(252.,device=device))).item(),
        'Corr':corr,'Corr val':corrv,'Pos mn':float((tm>0).float().mean()),'Neg mn':float((tm<0).float().mean()),
        'Pos mn val':float((vm>0).float().mean()),'Neg mn val':float((vm<0).float().mean())}
    even=tp[:2*(len(tp)//2):2].detach().cpu().numpy(); odd=tp[1:2*(len(tp)//2):2].detach().cpu().numpy()
    return pd.DataFrame([dt]),tw.detach().cpu().numpy(),even,odd

__all__=["Evaluation2LSTM"]
