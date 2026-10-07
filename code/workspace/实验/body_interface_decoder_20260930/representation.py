"""Image/prediction-only region descriptors; targets are a separate function."""
import numpy as np
from scipy import ndimage

from model import POINTS, TYPES, FEATURES, SUMMARY


def sample_group(ys,xs,cy,cx):
    if len(ys) == 0:
        return np.empty(0,dtype=np.int64)
    angle = np.arctan2(ys-cy,xs-cx)
    radius = (ys-cy)**2+(xs-cx)**2
    order = np.lexsort((xs,ys,radius,angle))
    positions = np.linspace(0,len(order)-1,min(POINTS,len(order)),dtype=np.int64)
    return order[positions]


def describe(image, probabilities, feature):
    """Inputs: grey HxW, probabilities 8xHxW, image features16xHxW."""
    h,w = image.shape
    if probabilities.shape != (8,h,w) or feature.shape != (16,h,w):
        raise ValueError("Bad descriptor input shape")
    prediction = probabilities.argmax(0).astype(np.uint8)
    oil = prediction >= 3
    labels,n = ndimage.label(oil,ndimage.generate_binary_structure(2,1))
    boxes = ndimage.find_objects(labels,max_label=n)
    kernel = np.array([[0,1,0],[1,0,1],[0,1,0]],dtype=np.float32)
    rock = ndimage.convolve((prediction==1).astype(np.float32),kernel,mode="constant",cval=0)/4
    water = ndimage.convolve((prediction==2).astype(np.float32),kernel,mode="constant",cval=0)/4
    padded = np.pad(oil.astype(np.float32),1)
    nx = .5*(padded[1:-1,2:]-padded[1:-1,:-2])
    ny = .5*(padded[2:,1:-1]-padded[:-2,1:-1])
    points = np.zeros((n,TYPES*POINTS,FEATURES),dtype=np.float32)
    valid = np.zeros((n,TYPES*POINTS),dtype=np.uint8)
    summaries = np.zeros((n,SUMMARY),dtype=np.float32)
    for i,box in enumerate(boxes):
        yy,xx = np.nonzero(labels[box]==i+1)
        yy,xx = yy+box[0].start,xx+box[1].start
        cy,cx = yy.mean(),xx.mean()
        height,width = box[0].stop-box[0].start,box[1].stop-box[1].start
        span = max(height,width)
        scale = max(span/2,1.)
        r,wc = rock[yy,xx]>0,water[yy,xx]>0
        summaries[i,:8] = probabilities[:,yy,xx].mean(1)
        summaries[i,8:] = [np.log1p(len(yy))/14,np.log1p(span)/8,len(yy)/(height*width),r.mean(),wc.mean()]
        for group,eligible in enumerate((np.ones(len(yy),dtype=bool),r,wc)):
            y,x = yy[eligible],xx[eligible]
            chosen = sample_group(y,x,cy,cx)
            y,x = y[chosen],x[chosen]
            if not len(y):
                continue
            values = np.concatenate((((x-cx)/scale)[None],((y-cy)/scale)[None],nx[y,x][None],ny[y,x][None],
                      rock[y,x][None],water[y,x][None],image[y,x][None],probabilities[:,y,x],feature[:,y,x]),0).T
            start = group*POINTS
            points[i,start:start+len(y)] = values
            valid[i,start:start+len(y)] = 1
    if not np.isfinite(points).all() or not np.isfinite(summaries).all():
        raise ValueError("Nonfinite descriptors")
    encoded = points.astype(np.float16)
    if not np.isfinite(encoded).all():
        raise ValueError("Descriptor overflows float16 cache")
    return {"points":encoded,"valid":valid,"summary":summaries}, labels, prediction


def targets(labels,prediction,ground_truth):
    """Label counts for loss/evaluation only. Never called by model.forward."""
    if not (labels.shape == prediction.shape == ground_truth.shape):
        raise ValueError("Unaligned labels")
    if ground_truth.min() < 0 or ground_truth.max() > 7:
        raise ValueError("Invalid target")
    n = int(labels.max())
    counts,outside = {},{}
    for grid,l,p,g in (("native",labels,prediction,ground_truth),
                       ("canonical512",labels[1::2,1::2],prediction[1::2,1::2],ground_truth[1::2,1::2])):
        counts[grid] = np.bincount((l.astype(np.int64)*8+g).ravel(),minlength=(n+1)*8).reshape(n+1,8)[1:].astype(np.uint32)
        keep = l==0
        outside[grid] = np.bincount((g[keep].astype(np.int64)*8+p[keep]),minlength=64).reshape(8,8)
        assert int(counts[grid].sum()+outside[grid].sum()) == g.size
    return counts,outside


def aggregate(counts,assigned,outside):
    h = np.asarray(outside,dtype=np.int64).copy()
    np.add.at(h.T,np.asarray(assigned,dtype=np.int64)+3,np.asarray(counts,dtype=np.int64))
    return h
