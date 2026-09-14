#!/usr/bin/env python3
"""Prepara pares R2→R5 registrados e ponderados para a busca de GANs.

Este estágio não olha R5 de 23/24. Os deslocamentos e pesos são calculados
somente nos pares supervisionados de 22/23; no alvo, QC usa apenas R2.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd

CONFIGS = {
    "identity": {"track": "A", "family": "baseline", "input_stages": ["R2"],
                 "model": "identity", "adversarial": False},
    "affine": {"track": "A", "family": "baseline", "input_stages": ["R2"],
               "model": "affine", "adversarial": False},
    "unet_l1": {"track": "A", "family": "deterministic", "input_stages": ["R2"],
                "model": "unet", "width": 32, "adversarial": False, "losses": ["L1"]},
    "unet_light": {"track": "A", "family": "gan", "input_stages": ["R2"],
                   "model": "unet", "width": 32, "adversarial": True,
                   "discriminators": 1, "losses": ["L1", "GAN"]},
    "resnet9": {"track": "A", "family": "gan", "input_stages": ["R2"],
                "model": "resnet", "width": 32, "blocks": 9, "adversarial": True,
                "discriminators": 1, "losses": ["L1", "GAN"]},
    "attention_unet": {"track": "A", "family": "gan", "input_stages": ["R2"],
                       "model": "attention_unet", "width": 32, "adversarial": True,
                       "discriminators": 1, "losses": ["L1", "GAN"]},
    "multiscale": {"track": "A", "family": "gan", "input_stages": ["R2"],
                   "model": "unet", "width": 32, "adversarial": True,
                   "discriminators": 2, "feature_matching": True,
                   "losses": ["L1", "GAN", "feature_matching"]},
    "wgangp": {"track": "A", "family": "gan_stability", "input_stages": ["R2"],
               "model": "unet", "width": 32, "adversarial": True,
               "gan_mode": "wgangp", "gradient_penalty": 10.0,
               "enabled_by_default": False,
               "trigger": "adversarial collapse or non-finite loss in at least two seeds",
               "losses": ["L1", "Wasserstein", "gradient_penalty"]},
    "history": {"track": "B", "family": "baseline", "input_stages": ["V8", "R2"],
                "model": "extratrees_direct", "adversarial": False},
    "convlstm_lite": {"track": "B", "family": "temporal", "input_stages": ["V8", "R2"],
                      "model": "convlstm", "width": 32, "adversarial": False,
                      "losses": ["L1"]},
    "simvp_lite": {"track": "B", "family": "temporal", "input_stages": ["V8", "R2"],
                   "model": "simvp", "width": 32, "adversarial": False,
                   "losses": ["L1"]},
}

def manifest(root: Path, stage: str) -> pd.DataFrame:
    d=pd.read_csv(root/'manifest.csv'); d=d[d.stage.str.upper()==stage].copy()
    if d.fid.duplicated().any() or 'mask' not in d: raise ValueError('manifesto requer máscaras e uma amostra por parcela')
    d['image']=[str((root/p).resolve()) for p in d.npy]; d['mask_file']=[str((root/p).resolve()) for p in d['mask']]
    return d

def shift_phase(a: np.ndarray,b: np.ndarray,mask:np.ndarray,max_shift=8):
    """Deslocamento R5→R2 por correlação de fase NIR dentro da máscara."""
    x,y=a[...,2]*mask,b[...,2]*mask
    corr=np.fft.ifft2(np.fft.fft2(x)*np.conj(np.fft.fft2(y))).real
    iy,ix=np.unravel_index(np.argmax(corr),corr.shape); h,w=x.shape
    dy=iy if iy<h//2 else iy-h; dx=ix if ix<w//2 else ix-w
    dy,dx=np.clip(dy,-max_shift,max_shift),np.clip(dx,-max_shift,max_shift)
    score=float(corr.max()/(np.linalg.norm(x)*np.linalg.norm(y)+1e-8))
    return int(dy),int(dx),score

def roll_zero(x,dy,dx):
    z=np.roll(x,(dy,dx),(0,1));
    if dy>0:z[:dy]=0
    if dy<0:z[dy:]=0
    if dx>0:z[:,:dx]=0
    if dx<0:z[:,dx:]=0
    return z

def quality(x,m):
    v=x[m>0]; ndvi=(v[:,2]-v[:,0])/(v[:,2]+v[:,0]+1e-8)
    sharp=np.var(np.diff(x[...,2],axis=0))+np.var(np.diff(x[...,2],axis=1))
    return {"mask_fraction":float(m.mean()),"ndvi":float(ndvi.mean()),"sharpness":float(sharp),"saturation":float((v>=.999).mean())}

def prepare(args):
    src_root,tgt_root,out=Path(args.source_stage4),Path(args.target_stage4),Path(args.out); out.mkdir(parents=True,exist_ok=True); (out/'source_r5_registered').mkdir(exist_ok=True)
    r2,r5=manifest(src_root,'R2'),manifest(src_root,'R5'); source=r2.merge(r5[['fid','image','mask_file']],on='fid',suffixes=('_r2','_r5'),validate='one_to_one')
    rows=[]
    for r in source.itertuples():
        a,b,m=np.load(r.image_r2),np.load(r.image_r5),np.load(r.mask_file_r2); dy,dx,c=shift_phase(a,b,m); aligned=roll_zero(b,dy,dx)*m[...,None]
        name=f'p{int(r.fid):03d}_r5_registered.npy'; np.save(out/'source_r5_registered'/name,aligned.astype('float32'))
        q=quality(a,m); # peso contínuo; nenhuma parcela é escolhida olhando biomassa.
        weight=np.clip((q['ndvi']/.8)*(q['sharpness']/(q['sharpness']+.001))*(1-q['saturation'])*max(c,0),.1,1.)
        rows.append({"fid":r.fid,"dose_n":r.dose_n,"bloco":r.bloco,"r2":r.image_r2,"mask":r.mask_file_r2,"r5_registered":str((out/'source_r5_registered'/name).resolve()),"shift_y":dy,"shift_x":dx,"phase_corr":c,"quality_weight":float(weight),**q})
    pd.DataFrame(rows).to_csv(out/'source_pairs_registered.csv',index=False)
    target=manifest(tgt_root,'R2'); target_rows=[]
    for r in target.itertuples(): target_rows.append({"fid":r.fid,"dose_n":r.dose_n,"bloco":r.bloco,"r2":r.image,"mask":r.mask_file,**quality(np.load(r.image),np.load(r.mask_file))})
    pd.DataFrame(target_rows).to_csv(out/'target_r2_qc.csv',index=False)
    (out/'architectures.json').write_text(json.dumps(CONFIGS,indent=2)+'\n')
    (out/'protocol.json').write_text(json.dumps({
        "selection": "leave-one-block-out 22/23; 23/24 locked",
        "primary_target": "Biomassa",
        "secondary_target": "Produtividade",
        "success": "delta_R2 >= 0.05; bootstrap lower95 > 0; Holm p < 0.05",
        "downstream": {"model": "ExtraTreesRegressor", "n_estimators": 500,
                       "min_samples_leaf": 2, "max_features": 1.0, "select_k": 8},
        "seeds": [7, 11, 23],
        "checkpoint_policy": "none",
        "mask_generated_before_features": True,
        "experimental_unit": "block+dose (24 independent field outcomes)",
        "scenarios": ["rs_only", "dose_only", "rs_plus_dose"],
        "target_r5_forbidden_before_final": True,
        "rankings_are_separate_by_track": True,
    },indent=2)+'\n')
    print(f'pares registrados: {len(rows)} | R2 alvo QC: {len(target_rows)} | configs: {len(CONFIGS)}')

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--source-stage4',required=True); p.add_argument('--target-stage4',required=True); p.add_argument('--out',required=True); prepare(p.parse_args())
if __name__=='__main__': main()
