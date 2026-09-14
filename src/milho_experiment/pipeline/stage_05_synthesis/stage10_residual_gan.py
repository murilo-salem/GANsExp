#!/usr/bin/env python3
"""GAN residual R2→R5 orientada à mudança de biomassa.

Usa manifests produzidos por ``stage4_phenology.py --reflectance-scale mosaic``.
R5 de 23/24 não é aceito neste CLI: o treino supervisionado é exclusivamente
22/23; 23/24 é previsto depois por ``predict``.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from .stage9_rfinal_forecast import TemporalGenerator, PatchDiscriminator, cond_channels, seed_all

def rows(stage4: Path, stage: str) -> pd.DataFrame:
    d=pd.read_csv(stage4/'manifest.csv'); d=d[d.stage.str.upper()==stage].copy()
    if 'mask' not in d or d.fid.duplicated().any(): raise ValueError('manifesto residual exige mask e uma linha por parcela/estágio')
    d['image_path']=[str((stage4/p).resolve()) for p in d.npy]; d['mask_path']=[str((stage4/p).resolve()) for p in d['mask']]
    return d

def source_pairs(stage4: Path, table: Path) -> pd.DataFrame:
    a,b=rows(stage4,'R2'),rows(stage4,'R5')
    x=a.merge(b[['fid','image_path','mask_path']],on='fid',suffixes=('_r2','_r5'),validate='one_to_one')
    t=pd.read_excel(table); t.Estagio=t.Estagio.astype(str).str.upper()
    y={(str(int(r.Dose_N)),int(r.Bloco),r.Estagio):float(r.Biomassa) for r in t.itertuples() if pd.notna(r.Biomassa)}
    def value(r,s): return y.get((str(int(float(r.dose_n))),int(r.bloco),s),np.nan)
    x['bio_r2']=[value(r,'R2') for r in x.itertuples()]; x['bio_r5']=[value(r,'R5') for r in x.itertuples()]
    return x.dropna(subset=['bio_r2','bio_r5']).reset_index(drop=True)

def condition(image, mask):
    return np.concatenate([image,cond_channels(image),mask[...,None]],-1).astype('float32')

class Pairs(Dataset):
    def __init__(self,d,delta_scale): self.d=d.reset_index(drop=True); self.delta_scale=delta_scale
    def __len__(self): return len(self.d)
    def __getitem__(self,i):
        r=self.d.iloc[i]; a=np.load(r.image_path_r2).astype('float32'); b=np.load(r.image_path_r5).astype('float32'); m=np.load(r.mask_path_r2).astype('float32')
        x=torch.from_numpy(condition(a,m)*2-1).permute(2,0,1); y=torch.from_numpy(b*2-1).permute(2,0,1); mask=torch.from_numpy(m)[None]
        return x,y,mask,torch.tensor((r.bio_r5-r.bio_r2)/self.delta_scale,dtype=torch.float32)

def masked_mean(v,m): return (v*m).sum()/(m.sum()*v.shape[1]+1e-8)
def idx(x):
    x=(x+1)/2; r,re,n=x[:,0:1],x[:,1:2],x[:,2:3]; return ((n-r)/(n+r+1e-7),(n-re)/(n+re+1e-7))
def gradient(x): return torch.abs(x[:,:,1:]-x[:,:,:-1]).mean()+torch.abs(x[:,:,:,1:]-x[:,:,:,:-1]).mean()

class ResidualGAN(nn.Module):
    def __init__(self):
        super().__init__(); self.g=TemporalGenerator(channels=8); self.delta_head=nn.Linear(128,1)
    def forward(self,x):
        # Limita a alteração para preservar R2; 0.25 é fração de reflectância.
        delta=.25*self.g(x); base=x[:,:3]; fake=torch.clamp(base+delta,-1,1)
        return fake, self.delta_head(self.g.features(x)).squeeze(1)

def train(args):
    seed_all(args.seed); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d=source_pairs(Path(args.source_stage4),Path(args.source_table)); tr=d[d.bloco!=args.validation_block]; va=d[d.bloco==args.validation_block]
    if tr.empty or va.empty: raise ValueError('bloco de validação inválido')
    scale=float((d.bio_r5-d.bio_r2).std()) or 1.; dl=DataLoader(Pairs(tr,scale),batch_size=args.batch_size,shuffle=True)
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); net=ResidualGAN().to(dev); disc=PatchDiscriminator(11).to(dev)
    og=torch.optim.Adam(net.parameters(),2e-4,betas=(.5,.999)); od=torch.optim.Adam(disc.parameters(),2e-4,betas=(.5,.999)); bce=nn.BCEWithLogitsLoss(); mse=nn.MSELoss(); best=1e9
    for ep in range(1,args.epochs+1):
        net.train()
        for x,y,m,db in dl:
            x,y,m,db=(z.to(dev) for z in (x,y,m,db)); fake,delta=net(x)
            real=disc(torch.cat([x,y],1)); bad=disc(torch.cat([x,fake.detach()],1)); lossd=.5*(bce(real,torch.ones_like(real))+bce(bad,torch.zeros_like(bad))); od.zero_grad(); lossd.backward(); od.step()
            adv=disc(torch.cat([x,fake],1)); ni,nj=idx(fake); ti,tj=idx(y)
            loss=bce(adv,torch.ones_like(adv))+100*masked_mean(torch.abs(fake-y),m)+10*(masked_mean(torch.abs(ni-ti),m)+masked_mean(torch.abs(nj-tj),m))+2*gradient((fake-y)*m)+mse(delta,db)
            og.zero_grad(); loss.backward(); og.step()
        net.eval()
        with torch.no_grad():
            v=[]
            for x,y,m,_ in DataLoader(Pairs(va,scale),batch_size=1): v.append(masked_mean(torch.abs(net(x.to(dev))[0]-y.to(dev)),m.to(dev)).item())
        if np.mean(v)<best: best=float(np.mean(v)); torch.save(net.state_dict(),out/'residual_gan.pt')
        print(f'epoch {ep}/{args.epochs} val_masked_l1={np.mean(v):.4f}')
    (out/'training.json').write_text(json.dumps({'best_source_val_masked_l1':best,'delta_biomass_scale':scale,'validation_block':args.validation_block},indent=2)+'\n')

def predict(args):
    stage4,out=Path(args.target_stage4),Path(args.out); out.mkdir(parents=True,exist_ok=True)
    d=rows(stage4,'R2'); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=ResidualGAN().to(dev); net.load_state_dict(torch.load(Path(args.checkpoint)/'residual_gan.pt',map_location=dev,weights_only=True)); net.eval(); result=[]
    for r in d.itertuples():
        image=np.load(r.image_path).astype('float32'); mask=np.load(r.mask_path).astype('float32')
        x=torch.from_numpy(condition(image,mask)*2-1).permute(2,0,1)[None].to(dev)
        with torch.no_grad(): fake,_=net(x)
        fake=((fake[0].permute(1,2,0).cpu().numpy()+1)/2*mask[...,None]).astype('float32')
        name=f'p{int(r.fid):03d}_R5_residual.npy'; np.save(out/name,fake)
        result.append({'fid':r.fid,'dose_n':r.dose_n,'bloco':r.bloco,'prediction':name})
    pd.DataFrame(result).to_csv(out/'predictions.csv',index=False); print(f'{len(result)} R5 residuais -> {out}')

def parser():
    p=argparse.ArgumentParser(description=__doc__); s=p.add_subparsers(dest='cmd',required=True)
    a=s.add_parser('train'); a.add_argument('--source-stage4',required=True); a.add_argument('--source-table',required=True); a.add_argument('--out',required=True); a.add_argument('--epochs',type=int,default=200); a.add_argument('--batch-size',type=int,default=2); a.add_argument('--validation-block',type=int,default=4); a.add_argument('--seed',type=int,default=42); a.set_defaults(func=train)
    a=s.add_parser('predict'); a.add_argument('--target-stage4',required=True); a.add_argument('--checkpoint',required=True); a.add_argument('--out',required=True); a.set_defaults(func=predict)
    return p
if __name__=='__main__':
    a=parser().parse_args(); a.func(a)
