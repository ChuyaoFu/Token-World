"""Numerically stable Gaussian Fréchet distances, including rank-deficient pilots."""
import numpy as np
from scipy import linalg


def distance(x,y,ddof=1):
 x=np.asarray(x,dtype=np.float64);y=np.asarray(y,dtype=np.float64)
 assert x.ndim==y.ndim==2 and x.shape[1]==y.shape[1] and min(len(x),len(y))>ddof
 assert np.isfinite(x).all() and np.isfinite(y).all()
 mx=x.mean(0);my=y.mean(0);a=(x-mx)/np.sqrt(len(x)-ddof);b=(y-my)/np.sqrt(len(y)-ddof)
 mean=float(np.square(mx-my).sum());ta=float(np.square(a).sum());tb=float(np.square(b).sum())
 if max(len(a),len(b))<x.shape[1]:
  cross=float(linalg.svdvals(a@b.T).sum());solver='equivalent low-rank nuclear norm'
 else:
  ca=a.T@a;cb=b.T@b;ea,va=linalg.eigh((ca+ca.T)*.5,check_finite=False)
  tolerance=max(1,float(np.abs(ea).max()))*1e-10
  if ea.min() < -tolerance:raise FloatingPointError(f'Non-PSD covariance: {ea.min()}')
  root=(va*np.sqrt(np.maximum(ea,0)))@va.T
  product=root@cb@root;ep=linalg.eigvalsh((product+product.T)*.5,check_finite=False)
  if ep.min() < -max(1,float(np.abs(ep).max()))*1e-8:raise FloatingPointError('Non-PSD covariance product')
  cross=float(np.sqrt(np.maximum(ep,0)).sum());solver='symmetric PSD eigendecomposition'
 result=mean+ta+tb-2*cross
 if result < -1e-6*max(1,ta+tb):raise FloatingPointError(f'Negative Fréchet distance {result}')
 return dict(value=max(0.,float(result)),feature_dimension=x.shape[1],real_samples=len(x),generated_samples=len(y),covariance_ddof=ddof,solver=solver,rank_deficiency_possible=min(len(x),len(y))-1<x.shape[1])


def check():
 rng=np.random.default_rng(42);x=rng.normal(size=(80,8));y=rng.normal(size=(70,8))
 a=distance(x,y);ref=np.square(x.mean(0)-y.mean(0)).sum()+np.trace(np.cov(x,rowvar=False)+np.cov(y,rowvar=False)-2*linalg.sqrtm(np.cov(x,rowvar=False)@np.cov(y,rowvar=False))).real
 assert abs(a['value']-ref)<1e-8,(a,ref)
 assert distance(x,x)['value']<1e-8
 shift=np.arange(8)/7;assert abs(distance(x,x+shift)['value']-np.square(shift).sum())<1e-8
 low=rng.normal(size=(5,64));other=rng.normal(size=(7,64))
 assert abs(distance(low,other)['value']-distance(other,low)['value'])<1e-8
 assert distance(low,low)['value']<1e-8
 return dict(status='passed',checks=['agreement with scipy sqrtm on SPD covariances','identity distance zero','known translation distance','low-rank symmetry and identity'],maximum_SPD_reference_error=abs(a['value']-ref))
