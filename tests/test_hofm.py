import unittest
from time import time
import itertools 
import torch

import sys
sys.path.insert(0,"../src/")

######################################################
## Computation of the HOFM formula (forward pass)   ##
######################################################

if (True):
    ## Exact formula for any m
    def exact_anova_kernel(x, w, m):
        st=time()
        out = 0
        xx = torch.tile(x.T, (w.size(dim=1),1,1))
        ww = torch.tile(w, (x.size(dim=0),1,1))
        ww = ww.permute(*torch.arange(ww.ndim - 1, -1, -1))
        wx = torch.mul(xx, ww)
        wx = wx.permute(*torch.arange(wx.ndim - 1, -1, -1))  ## n x F x d
        for t in range(2,m+1):
            Akt = 0 
            Fs = itertools.combinations(range(w.size(dim=0)), t)
            for Fc in Fs:                                        
                Akt += wx[:,list(Fc),:].prod(dim=1,keepdim=False).sum(dim=1,keepdim=True) 
            out += Akt
            #Aexact[:,t-2] = Akt.reshape(n)
        print("Exact formula for m=%d d=%d: %.9f sec\n" % (m, x.size(dim=0), time()-st))
        return out

    ## Fast formula for order m=2  
    def fast_anova_kernel(x, w, m):
        assert m==2
        st=time()
        out_pair1 = torch.matmul(x, w).pow(2).sum(1, keepdim=True) 
        out_pair2 = torch.matmul(x.pow(2), w.pow(2)).sum(1, keepdim=True)
        out = 0.5*(out_pair1-out_pair2)
        print("Fast formula for m=2 d=%d: %.9f sec\n" % (x.size(dim=0),time()-st))
        return out
 
    ## Dynamic programming algorithm (Algorithm 1 from Blondel et al., 2016) for any m     
    def dp_anova_kernel(x, w, m):
        st=time()
        out = 0
        n, F = x.size()
        _, d = w.size()
        sparse_F = torch.linspace(0,F-1,F)[x.abs().sum(0)>0] ## only features f such that there exists i, x[i,f]!=0
        FF = sparse_F.size(dim=0)
        for s in range(d):
            A = torch.zeros(n, FF+1, m+1)
            A[:,:,0] = 1
            for t in range(2, m+1):
                ## Algorithm 1 to compute ANOVA kernel of order t on column s of w   
                ## https://arxiv.org/pdf/1607.07195.pdf
                for i in range(t-1,t+1):       
                    for j in range(i, FF+1):
                        A[:,j,i] = A[:,j-1,i] + w[int(sparse_F[j-1]),s]*(A[:,j-1,i-1] * x[:,int(sparse_F[j-1])])                                      
                    #out += A[:, FF, t].reshape(n, 1)
            out += A[:,FF,2:].sum(-1, keepdim=True)
        print("DP formula for m=%d d=%d: %.9f sec\n" % (m,x.size(dim=0), time()-st))
        return out
        
######################################################
## Testing the computation of the HOFM formula      ##
######################################################

class TestHOFM(unittest.TestCase):

    def test_order2(self):
        print("_"*27)
        x, w = torch.normal(0,1,size=(5,600)).round(), torch.normal(0,1,size=(600,50))
        self.assertTrue((((fast_anova_kernel(x,w,2)-dp_anova_kernel(x,w,2))).abs()<1e-1).all())
        self.assertTrue((((fast_anova_kernel(x,w,2)-exact_anova_kernel(x,w,2))).abs()<1e-1).all())
        
    def test_order5(self):
        print("_"*27)
        x, w = torch.normal(0,1,size=(100,20)).round(), torch.normal(0,1,size=(20,50))
        self.assertTrue((((dp_anova_kernel(x,w,5)-exact_anova_kernel(x,w,5))).abs()<1e-1).all())

    def test_order2_realistic(self):
        print("_"*27)
        x, w = torch.normal(0,1,size=(2000,12000)).round(), torch.normal(0,1,size=(12000,50)) ## realistic numbers
        x1 = fast_anova_kernel(x,w,2)
        x2 = dp_anova_kernel(x,w,2)
        #print((x1,x2))
        self.assertTrue((((x1-x2)).abs()<1).all())
        
if __name__ == '__main__':
    unittest.main()
