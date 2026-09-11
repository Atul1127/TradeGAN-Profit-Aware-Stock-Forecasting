"""Objective formulas and gradient-balance helpers for TradeGAN."""
import torch

def trading_terms(prediction,real,tanh_coefficient=100.0):
    pnl_samples=torch.tanh(tanh_coefficient*prediction)*real; pnl=pnl_samples.mean(); mse=torch.norm(prediction-real)**2/prediction.shape[0]; std=pnl_samples.std(); sr=pnl/std.clamp_min(torch.finfo(pnl.dtype).eps); return pnl,mse,sr,std

def gan_loss_bce(bce): return bce
def gan_loss_pnl(bce,alpha,pnl): return bce-alpha*pnl
def gan_loss_pnl_mse(bce,alpha,pnl,beta,mse): return bce-alpha*pnl+beta*mse
def gan_loss_pnl_mse_sr(bce,alpha,pnl,beta,mse,gamma,sr): return bce-alpha*pnl+beta*mse-gamma*sr
def gan_loss_pnl_mse_std(bce,alpha,pnl,beta,mse,delta,std): return bce-alpha*pnl+beta*mse+delta*std
def gan_loss_pnl_sr(bce,alpha,pnl,gamma,sr): return bce-alpha*pnl-gamma*sr
def gan_loss_mse(bce,beta,mse): return bce+beta*mse
def gan_loss_sr(bce,gamma,sr): return bce-gamma*sr
def gan_loss_sr_mse(bce,beta,mse,gamma,sr): return bce+beta*mse-gamma*sr
def gan_loss_pnl_std(bce,alpha,pnl,delta,std): return bce-alpha*pnl+delta*std
def lstm_loss_mse(mse): return mse
def lstm_loss_pnl(mse,alpha,pnl): return mse-alpha*pnl
def lstm_loss_pnl_std(mse,alpha,pnl,delta,std): return mse-alpha*pnl+delta*std
def lstm_loss_pnl_sr(mse,alpha,pnl,gamma,sr): return mse-alpha*pnl-gamma*sr
def lstm_loss_sr(mse,gamma,sr): return mse-gamma*sr
def lstm_loss_std(mse,delta,std): return mse+delta*std

def _grad_norm(module):
    total=torch.zeros((),device=next(module.parameters()).device)
    for p in module.parameters():
        if p.grad is not None: total=total+p.grad.detach().norm(2)**2
    return total.sqrt()

def _safe_ratio(num,den): return num/den.clamp_min(torch.finfo(num.dtype).eps)

def GradientCheck(ticker,gen,disc,gen_opt,disc_opt,criterion,n_epochs,train_data,batch_size,hid_d,hid_g,z_dim,lr_d=0.0001,lr_g=0.0001,h=1,l=10,pred=1,diter=1,tanh_coeff=100,device="cpu",plot=False):
    del ticker,lr_d,lr_g,h,plot
    if n_epochs<0 or batch_size<=0 or diter<=0: raise ValueError("invalid gradient-check parameters")
    n=train_data.shape[0]
    if n==0: raise ValueError("train_data must not be empty")
    vals=[[] for _ in range(5)]; gen.train(); disc.train()
    for _ in range(n_epochs):
        shuffled=train_data[torch.randperm(n,device=train_data.device)]
        for start in range(0,n,batch_size):
            batch=shuffled[start:start+batch_size]; cur=batch.shape[0]; c=batch[:,:l].unsqueeze(0).to(device=device,dtype=torch.float); r=batch[:,l:l+pred].unsqueeze(0).to(device=device,dtype=torch.float)
            h0d=torch.zeros((1,cur,hid_d),device=device); c0d=torch.zeros_like(h0d); h0g=torch.zeros((1,cur,hid_g),device=device); c0g=torch.zeros_like(h0g)
            disc_opt.zero_grad(set_to_none=True); fake=gen(torch.randn((1,cur,z_dim),device=device),c,h0g,c0g); fp=disc(torch.cat((c,fake.detach()),-1),h0d,c0d); rp=disc(torch.cat((c,r),-1),h0d,c0d); dl=(criterion(fp,torch.zeros_like(fp))+criterion(rp,torch.ones_like(rp)))/2; dl.backward(); disc_opt.step()
            fake=gen(torch.randn((1,cur,z_dim),device=device),c,h0g,c0g); fp=disc(torch.cat((c,fake),-1),h0d,c0d); p=fake.squeeze(0).squeeze(-1); t=r.squeeze(0).squeeze(-1); ps=torch.tanh(tanh_coeff*p)*t
            objectives=[criterion(fp,torch.ones_like(fp)),ps.mean(),torch.norm(p-t)**2/cur,ps.mean()/ps.std().clamp_min(torch.finfo(ps.dtype).eps),ps.std()]
            for i,v in enumerate(objectives): gen_opt.zero_grad(set_to_none=True); v.backward(retain_graph=True); vals[i].append(_grad_norm(gen).detach())
            gen_opt.step()
    if not vals[0]:
        z=torch.ones((),device=device); return gen,disc,gen_opt,disc_opt,z,z,z,z
    B,P,M,S,D=map(torch.stack,vals); return gen,disc,gen_opt,disc_opt,_safe_ratio(B,P).mean(),_safe_ratio(B,M).mean(),_safe_ratio(B,S).mean(),_safe_ratio(B,D).mean()

def GradientCheckLSTM(ticker,gen,gen_opt,n_epochs,train_data,batch_size,hid_d,hid_g,z_dim,lr_d=0.0001,lr_g=0.0001,h=1,l=10,pred=1,diter=1,tanh_coeff=100,device="cpu",plot=False):
    del ticker,hid_d,z_dim,lr_d,lr_g,h,diter,plot
    if n_epochs<0 or batch_size<=0: raise ValueError("invalid gradient-check parameters")
    n=train_data.shape[0]
    if n==0: raise ValueError("train_data must not be empty")
    vals=[[] for _ in range(4)]; gen.train()
    for _ in range(n_epochs):
        shuffled=train_data[torch.randperm(n,device=train_data.device)]
        for start in range(0,n,batch_size):
            batch=shuffled[start:start+batch_size]; cur=batch.shape[0]; c=batch[:,:l].unsqueeze(0).to(device=device,dtype=torch.float); r=batch[:,l:l+pred].unsqueeze(0).to(device=device,dtype=torch.float); h0=torch.zeros((1,cur,hid_g),device=device); f=gen(c,h0,torch.zeros_like(h0)); p=f.squeeze(0).squeeze(-1); t=r.squeeze(0).squeeze(-1); ps=torch.tanh(tanh_coeff*p)*t; objectives=[ps.mean(),torch.norm(p-t)**2/cur,ps.mean()/ps.std().clamp_min(torch.finfo(ps.dtype).eps),ps.std()]
            for i,v in enumerate(objectives): gen_opt.zero_grad(set_to_none=True); v.backward(retain_graph=True); vals[i].append(_grad_norm(gen).detach())
            gen_opt.step()
    if not vals[0]:
        z=torch.ones((),device=device); return gen,gen_opt,z,z*0,z,z
    P,M,S,D=map(torch.stack,vals); return gen,gen_opt,_safe_ratio(M,P).mean(),torch.zeros((),device=device),_safe_ratio(M,S).mean(),_safe_ratio(M,D).mean()

__all__=["trading_terms","GradientCheck","GradientCheckLSTM","gan_loss_bce","gan_loss_pnl","gan_loss_pnl_mse","gan_loss_pnl_mse_sr","gan_loss_pnl_mse_std","gan_loss_pnl_sr","gan_loss_mse","gan_loss_sr","gan_loss_sr_mse","gan_loss_pnl_std","lstm_loss_mse","lstm_loss_pnl","lstm_loss_pnl_std","lstm_loss_pnl_sr","lstm_loss_sr","lstm_loss_std"]
