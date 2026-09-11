"""GAN evaluation without legacy dependencies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch


def _stats(samples, real):
    mean = samples.mean(0); real = real.squeeze()
    rmse=torch.sqrt(torch.mean((mean-real)**2)); mae=torch.mean(torch.abs(mean-real))
    pnl=10000*torch.sign(mean)*real
    weekly=pnl[:2*(len(pnl)//2)].reshape(-1,2).sum(1)
    sr=weekly.mean()/weekly.std().clamp_min(torch.finfo(weekly.dtype).eps)
    return mean,rmse,mae,pnl,weekly,sr


def Evaluation2(ticker,freq,gen,test_data,val_data,h,l,pred,hid_d,hid_g,z_dim,lrg,lrd,n_epochs,losstype,sr_val,device,plotsloc,f_name,plot=False,mc_samples=1000):
    del freq,hid_d,lrg,lrd,n_epochs,sr_val,plotsloc,f_name,plot
    if mc_samples<=0: raise ValueError("mc_samples must be positive")
    def evaluate(data):
        n=data.shape[0]
        cond=data[:,:l].unsqueeze(0).to(device=device,dtype=torch.float)
        h0=torch.zeros((1,n,hid_g),device=device); c0=torch.zeros_like(h0)
        draws=[]
        gen.eval()
        with torch.no_grad():
            for _ in range(mc_samples):
                noise=torch.randn((1,n,z_dim),device=device)
                draws.append(gen(noise,cond,h0,c0).squeeze(0).squeeze(-1))
        samples=torch.stack(draws)
        real=data[:,-1].to(device=device,dtype=torch.float)
        return _stats(samples,real)
    tm,trm,tma,tp,tw,tsr=evaluate(test_data); vm,vrm,vma,vp,vw,vsr=evaluate(val_data)
    even=tp[:2*(len(tp)//2):2].detach().cpu().numpy(); odd=tp[1:2*(len(tp)//2):2].detach().cpu().numpy()
    dt={'lrd':lrd,'lrg':lrg,'type':losstype,'epochs':n_epochs,'ticker':ticker,'hid_g':hid_g,'hid_d':hid_d,
        'RMSE':trm.item(),'MAE':tma.item(),'PnL_w':tw.mean().item(),'SR_w scaled':(tsr*torch.sqrt(torch.tensor(252.,device=device))).item(),
        'RMSE val':vrm.item(),'MAE val':vma.item(),'PnL_w val':vw.mean().item(),'SR_w scaled val':(vsr*torch.sqrt(torch.tensor(252.,device=device))).item(),
        'Corr':np.corrcoef(tm.cpu().numpy(),test_data[:,-1].cpu().numpy())[0,1],
        'Corr val':np.corrcoef(vm.cpu().numpy(),val_data[:,-1].cpu().numpy())[0,1],
        'Pos mn':float((tm>0).float().mean()),'Neg mn':float((tm<0).float().mean()),
        'Pos mn val':float((vm>0).float().mean()),'Neg mn val':float((vm<0).float().mean()),
        'narrow dist':bool(samples_std:=False),'narrow means dist':False}
    return pd.DataFrame([dt]),tw.detach().cpu().numpy(),even,odd,tm.detach().cpu().numpy(),test_data[:,-1].cpu().numpy(),tm.detach().cpu().numpy(),test_data[:,-1].cpu().numpy()


def Evaluation3(tickers,freq,gen,test,val,h,l,pred,hid_d,hid_g,z_dim,lrg,lrd,n_epochs,losstype,sr_val,device,plotsloc,f_name,plot=False):
    rows=[]; pnlt=[]; pnlv=[]
    for i,ticker in enumerate(tickers):
        df,*rest=Evaluation2(ticker,freq,gen,test[i],val[i],h,l,pred,hid_d,hid_g,z_dim,lrg,lrd,n_epochs,losstype,sr_val,device,plotsloc,f_name,plot)
        rows.append(df.iloc[0]); pnlt.append(rest[0]); pnlv.append(rest[2])
    return pd.DataFrame(rows).reset_index(drop=True),np.sum(pnlt,axis=0),np.sum(pnlv,axis=0),None,None

__all__=["Evaluation2","Evaluation3"]
