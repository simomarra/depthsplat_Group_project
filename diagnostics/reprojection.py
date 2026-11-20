"""NumPy forward reprojection: normalized pixel centers, c2w poses, camera-Z depth.

Same normalized grid convention as src.geometry.projection.sample_image_grid.
No hole filling, no Gaussian attributes. Z-buffering cannot resolve all true occlusions.
"""
import numpy as np


def project(depth, rgb, ka, ta, kb, tb, near, far):
    h,w=depth.shape
    if rgb.shape!=(h,w,3):raise ValueError('RGB/depth size mismatch.')
    x=(np.arange(w,dtype=np.float32)+.5)/w
    y=(np.arange(h,dtype=np.float32)+.5)/h
    xx,yy=np.meshgrid(x,y,indexing='xy')
    xy=np.stack((xx,yy),axis=-1).astype(np.float64)
    ids=np.flatnonzero(np.isfinite(depth.ravel()) & (depth.ravel()>0))
    uv1=np.column_stack((xy.reshape(-1,2)[ids],np.ones(len(ids))))
    points=(uv1@np.linalg.inv(ka).T)*depth.ravel()[ids,None]
    relative=np.linalg.inv(tb)@ta
    points=points@relative[:3,:3].T+relative[:3,3]
    z=points[:,2]
    valid=np.isfinite(points).all(1)&(z>0)&(z>=near*(1-1e-6))&(z<=far*(1+1e-6))
    points,z,ids=points[valid],z[valid],ids[valid]
    uvh=points@kb.T
    with np.errstate(divide='ignore',invalid='ignore'):
        uv=uvh[:,:2]/uvh[:,2:]
    # Filter in floating point BEFORE integer conversion to avoid overflow.
    valid=np.isfinite(uv).all(1)&(uv[:,0]>=0)&(uv[:,0]<1)&(uv[:,1]>=0)&(uv[:,1]<1)
    uv,z,ids=uv[valid],z[valid],ids[valid]
    pixels=np.floor(uv*np.array([w,h])).astype(np.int64)
    pix=pixels[:,1]*w+pixels[:,0]
    order=np.lexsort((ids,z,pix))
    pix,z,ids=pix[order],z[order],ids[order]
    keep=np.r_[True,pix[1:]!=pix[:-1]] if len(pix) else np.zeros(0,bool)
    pix,z,ids=pix[keep],z[keep],ids[keep]
    image=np.zeros((h*w,3),np.float32)
    buffer=np.full(h*w,np.inf)
    source=np.full(h*w,-1,np.int64)
    image[pix]=rgb.reshape(-1,3)[ids];buffer[pix]=z;source[pix]=ids
    return image.reshape(h,w,3),buffer.reshape(h,w),source.reshape(h,w)


def combine(warps):
    images=np.stack([v[0] for v in warps]);depths=np.stack([v[1] for v in warps])
    choices=depths.argmin(0);yy,xx=np.indices(choices.shape)
    return images[choices,yy,xx],depths[choices,yy,xx]


def compare(a,b,gt):
    ma,mb=np.isfinite(a[1]),np.isfinite(b[1]);common=ma&mb
    ea=np.abs(a[0]-gt).mean(-1);eb=np.abs(b[0]-gt).mean(-1)
    scores={'coverage_baseline':float(ma.mean()),'coverage_secondary':float(mb.mean()),
        'common_coverage':float(common.mean()),
        'mae_baseline_common':float(ea[common].mean()) if common.any() else None,
        'mae_secondary_common':float(eb[common].mean()) if common.any() else None}
    return scores,common,ea,eb
