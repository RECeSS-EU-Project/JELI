#coding: utf-8

import torch
import torch.nn as nn
from torch.linalg import solve
from tqdm import tqdm
import random
import os
import numpy as np
from copy import deepcopy
from scipy.sparse import coo_array
import pathlib
from pykeen.models import ERModel
from pykeen.training import SLCWATrainingLoop
from pykeen.sampling import negative_sampler_resolver
from pykeen.nn.functional import mure_interaction
from sklearn.metrics import pairwise_distances

from stanscofi.training_testing import random_cv_split
from sklearn.model_selection import StratifiedKFold
from torch_geometric.utils import negative_sampling

from pykeen.nn.modules import MuREInteraction
from pykeen.pipeline.api import set_random_seed, _build_model_helper, resolve_device, optimizer_resolver, training_loop_resolver, stopper_resolver
from pykeen.utils import upgrade_to_sequence
from typing import Sequence
from joblib import Parallel, delayed, parallel_backend

import igraph as ig
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import normalize
from qnorm import quantile_normalize
        
###################################################################################
## Redundant Structured HOFM                                                     ##
###################################################################################
class RHOFM(nn.Module):
    '''
    Redundant / Inhomogeneous anova kernel with structure Higher Order Factorization Machine

    ...

    Parameters
    ----------
    d : int
        dimension
    order : int
    	order
    structure : function or "linear"
    	the dependency on feature embeddings
    cuda_on : bool
    	use cuda if True

    Attributes
    ----------
    theta : torch.nn.Parameter of size (order+1,1)
    	the coefficients of terms of order 0, 2, ..., order
    theta1 : torch.nn.Parameter of size (d, 1)
        the coefficient of term of order 1
    structure : function
        the dependency on feature embeddings
    is_linear : bool
    	is the dependency linear (fasten computations)

    Methods
    -------
    __init__(d, order, structure, cuda_on)
        Initializes the RHOFM
    forward(item, user, embs, lbd)
        Computes the HOFM function on vector [item, user] and embeddings embs
    '''
    def __init__(self, d, order, structure, frozen, cuda_on, random_state=1234):
        '''
        Creates an instance of RHOFM

        ...

        Parameters
        ----------
    	d : int
            dimension
    	order : int
            order
    	structure : function or "linear"
            the dependency on feature embeddings
    	cuda_on : bool
            use cuda if True
        '''
        super().__init__()
        assert order>=1
        if (frozen):
            self.theta2 = to_cuda(torch.ones(order-1, 1), cuda_on)
            self.theta1 = to_cuda(torch.ones(d, 1), cuda_on)
            self.theta0 = to_cuda(torch.ones(1, 1), cuda_on) ##
        else:
            self.theta2 = to_cuda(nn.Parameter(torch.randn(order-1, 1), requires_grad=True), cuda_on)
            self.theta1 = to_cuda(nn.Parameter(torch.randn(d, 1), requires_grad=True), cuda_on)
            self.theta0 = to_cuda(nn.Parameter(torch.randn(1, 1), requires_grad=True), cuda_on) ##
        if (structure=="linear"):
            self.structure = lambda x, w : torch.matmul(x, w)
            self.is_linear = True
        else:
            self.structure = structure
            self.is_linear = False
        self.cuda_on = cuda_on
        self.embeddings = None
        self.lbd=0.001
        self.seed_everything(random_state)
        
    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        #torch.use_deterministic_algorithms(True)
        
    def forward(self, inp):
        '''
        Outputs properly formatted scores (not necessarily in [0,1]!) from the fitted model on test_dataset. Internally calls model_predict() then reformats the scores

        ...

        Parameters
        ----------
        test_dataset : stanscofi.Dataset
            dataset on which predictions should be made

        Returns
        ----------
        scores : COO-array of shape (n_items, n_users)
            sparse matrix in COOrdinate format, with nonzero values corresponding to predictions on available pairs in the dataset
        '''
        item, user, embs = inp
        if (self.is_linear): ## ww = embs
            w = to_cuda(broadcast_cat([embs]*2, dim=0), self.cuda_on)
        ## factorize the structure if it is not linear
        ## find w st. self.structure(x, embs) ~ torch.matmul(x, w)
        else:
            x_stacked = broadcast_cat([item, user], dim=0)
            fx = self.structure(x_stacked, embs)
            A = torch.matmul(x_stacked.T, x_stacked)+self.lbd*torch.eye(x_stacked.size(dim=-1))
            B = torch.matmul(x_stacked.T, fx)
            ww = solve(A,B) ## if self.structure(x_stacked, embs) is differentiable in embs, so is torch.matmul(x_stacked, ww)
            w = to_cuda(broadcast_cat([ww]*2, dim=0), self.cuda_on)
        m = self.theta2.size(dim=0)+1
        
        #print(("range pair", item.min(), item.max(), user.min(), user.max()))
        
        x = to_cuda(broadcast_cat([item, user], dim=-1), self.cuda_on) ## concatenate user and item feature vectors
        
        #print(("range param", w.min(), w.max(), x.min(), x.max()))
        out = self.theta0 + torch.matmul(torch.matmul(x, w), self.theta1)  
        if (m==2): ## fast approach for order 2
            out_pair1 = torch.matmul(x, w).pow(2).sum(1, keepdim=True) 
            out_pair2 = torch.matmul(x.pow(2), w.pow(2)).sum(1, keepdim=True)
            out += 0.5*self.theta2*(out_pair1-out_pair2)
        elif (m>2): ## dynamic programming algorithm for order > 2
            n, F = x.size()
            _, d = w.size()
            sparse_F = to_cuda(torch.linspace(0,F-1,F), self.cuda_on)[x.abs().sum(0)>0] ## only features f such that there exists i, x[i,f]!=0
            FF = sparse_F.size(dim=0)
            for s in range(d):
                A = to_cuda(torch.zeros(n, FF+1, m+1), self.cuda_on)
                A[:,:,0] = 1
                for t in range(2, m+1):
                    ## Algorithm 1 to compute ANOVA kernel of order t on column s of w   
                    ## https://arxiv.org/pdf/1607.07195.pdf
                    for i in range(t-1,t+1):       
                        for j in range(i, FF+1):
                            A[:,j,i] = A[:,j-1,i] + w[int(sparse_F[j-1]),s]*(A[:,j-1,i-1] * x[:,int(sparse_F[j-1])])
                out += torch.matmul( A[:, FF, 2:], self.theta2 )#.reshape(n, 1)
        return out
     
    ############################################
    ### Separate learning of FM coefficients ###
    ############################################
    
    def sample(self, dataset, batch_size, stype="negative", batch_seed=1234, num_neg_samples=3, method="sparse", force_undirected=False):  
        assert stype in ["uniform", "negative", "deterministic"]
        #stype = "deterministic"
        if (stype == "deterministic"): ## for testing purposes
            n = len(dataset.folds.data)
            batch_size = min(batch_size, n)
            nbatches = n//batch_size+int(n%batch_size!=0)
            M = len(dataset.folds.row)
            batch_folds = [coo_array((np.ones(n), (dataset.folds.row[i:i+n], dataset.folds.col[i:i+n])), shape=dataset.folds.shape) for i in range(0, M, n)]
            return batch_folds
        elif (stype == "uniform"):  
            n = len(dataset.folds.data)
            batch_size = min(batch_size, n)
            nbatches = n//batch_size+int(n%batch_size!=0)
            cv_generator = StratifiedKFold(n_splits=nbatches, shuffle=True, random_state=batch_seed)
            batch_folds, _ = random_cv_split(dataset, cv_generator, metric="cosine")
            return [b for _, b in batch_folds]
        elif (stype == "negative"):  
            self.num_neg_samples = num_neg_samples
            pos_folds = dataset.ratings.toarray()
            pos_folds = torch.as_tensor(np.argwhere(pos_folds).T)
            batch_size = min(batch_size, pos_folds.shape[1])
            nbatches = pos_folds.shape[1]//batch_size+int(pos_folds.shape[1]%batch_size!=0)
            neg_folds = negative_sampling(pos_folds, num_nodes=dataset.ratings.shape, num_neg_samples=num_neg_samples, method=method, force_undirected=force_undirected)
            batch_folds = []
            pos_folds, neg_folds = pos_folds.numpy(), neg_folds.numpy()
            for batch in range(nbatches):
                pfolds = pos_folds[:,batch*batch_size//2:(batch+1)*batch_size//2]
                nfolds = neg_folds[:,batch*batch_size//2:(batch+1)*batch_size//2]
                data = np.array([1]*(pfolds.shape[1]+nfolds.shape[1]))
                row = np.concatenate((pfolds[0,:], nfolds[0,:]), axis=0)
                col = np.concatenate((pfolds[1,:], nfolds[1,:]), axis=0)
                batch_folds.append(coo_array((data, (row, col)), shape=dataset.ratings.shape))
            return batch_folds ## outputs *ONE* batch if too few positive
        else:
            raise NotImplemented
       
     ## SGD and CD pipelines in https://arxiv.org/pdf/1607.07195.pdf
    def fit(self, dataset, embeddings=None, n_epochs=25, batch_size=100, optimizer_class=torch.optim.AdamW, loss="MarginRankingLoss", opt_params={'lr': 0.001, "weight_decay": 0.01}, early_stop=0, random_seed=1234):
        assert np.isfinite(dataset.items.toarray()).all()
        assert np.isfinite(dataset.users.toarray()).all()
        assert dataset.nitem_features == dataset.nuser_features
        F = dataset.nitem_features
        if (embeddings is None):
        	self.embeddings = to_cuda(nn.Parameter(torch.randn(self.theta1.size(dim=0), F), requires_grad=True), self.cuda_on)
        else:
        	self.embeddings = to_cuda(torch.Tensor(embeddings), self.cuda_on)
        params = [x for x in [self.theta0, self.theta1, self.theta2] if ('torch.nn.parameter.Parameter' in str(type(x)))]+([self.embeddings] if (embeddings is None) else [])
        optimizer = optimizer_class(tuple(params), **opt_params)
        train_losses = []
        #scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [cyclical_lr(10000)])
        old_epoch_loss, early_stop_counter = float("inf"), 0
        with tqdm(total=n_epochs * len(dataset.folds.data)) as pbar:
            for epoch in range(n_epochs):
                epoch_loss, epoch_train_losses, n_epoch = 0, [], 0
                #for batch_id, batch_fold in (pbar := tqdm(enumerate(self.sample(dataset, batch_size, batch_seed=random_seed+epoch)))):
                for batch_id, batch_fold in enumerate(self.sample(dataset, batch_size, batch_seed=random_seed+epoch)):
                    batch_y = to_cuda(torch.LongTensor(dataset.ratings.toarray()[batch_fold.row,batch_fold.col].ravel()), self.cuda_on)
                    batch_y[batch_y<0] = 0
                    items = dataset.items.toarray().T[batch_fold.row,:]
                    users = dataset.users.toarray().T[batch_fold.col]
                    batch_item, batch_user = [to_cuda(torch.Tensor(X), self.cuda_on) for X in [items, users]]
                    optimizer.zero_grad()
                    out = self((batch_item, batch_user, self.embeddings))
                    try:
                    	out_cross = torch.cat((out, -out), dim=1) ## cross-enthropy
                    	loss_epoch = eval("nn."+loss)()(out_cross, batch_y)
                    except:
                    	assert self.num_neg_samples>0
                    	out_neg = out[batch_y!=1]  ## pairwise
                    	out_pos = out[batch_y==1].repeat((out_neg.size(dim=0)//out[batch_y==1].size(dim=0)+1,1))[:out_neg.size(dim=0)]
                    	target = torch.ones(out_neg.size())
                    	#print((out[batch_y==1].size(), self.num_neg_samples, out_pos.size(), target.size(), out_neg.size()))
                    	loss_epoch = eval("nn."+loss)()(out_pos, out_neg, target)
                    with torch.set_grad_enabled(True):
                        #with torch.autograd.detect_anomaly():
                        loss_epoch.backward()
                        #scheduler.step()
                        optimizer.step()
                    n_epoch += batch_item.size(0)
                    epoch_loss += loss_epoch.item()*batch_item.size(0)
                    pbar.set_description("Batch #%d (epoch %d): loss %f (prev %f)" % (batch_id+1, epoch+1, epoch_loss/n_epoch, np.nan if (len(epoch_train_losses)==0) else epoch_train_losses[-1]))
                    epoch_train_losses.append(epoch_loss/n_epoch)
                train_losses.append(epoch_train_losses)
                if (old_epoch_loss<epoch_loss):
                    early_stop_counter += 1
                    old_epoch_loss = epoch_loss
                if ((early_stop>0) and (early_stop_counter>early_stop)):
                    break
        return train_losses
        
    def predict_proba(self, dts, default_zero_val=1e-31):
        assert self.embeddings is not None
        items = torch.Tensor(dts.items.toarray().T[dts.folds.row,:])
        users = torch.Tensor(dts.users.toarray().T[dts.folds.col])
        scores_data = torch.sigmoid(self((items, users, self.embeddings))).detach().cpu().numpy().flatten()
        scores = coo_array((scores_data, (dts.folds.row, dts.folds.col)), shape=dts.ratings.shape)
        scores = scores.toarray()
        default_val = min(default_zero_val, np.min(scores[scores!=0])/2 if ((scores!=0).any()) else default_zero_val)
        scores[(scores==0)&(dts.folds.toarray()==1)] = default_val
        scores = coo_array(coo_array(scores)*dts.folds)
        return scores
        
    def predict(self, scores, threshold=0.5):
        preds = coo_array((scores.toarray()!=0).astype(int)*((-1)**(scores.toarray()<=0.5)))
        return preds

#########################################################################
## Extending the MuRE interaction model with FM outputs                ##
#########################################################################
class MuRE_RHOFM(MuREInteraction):
	entity_shape = ("d", "", "")
	relation_shape = ("d", "d")
	func = mure_interaction
	def __init__(self, d, order, structure, kge, frozen=False, cuda_on=False, p_norm=2, random_seed=1234):
		super().__init__(p=p_norm,power_norm=False)
		self.seed_everything(seed=random_seed)
		self.FM = RHOFM(d, order, structure, frozen, cuda_on, random_seed)
		self.kge = kge
		self.cuda_on = cuda_on
		
	def seed_everything(self, seed):
		random.seed(seed)
		os.environ['PYTHONHASHSEED'] = str(seed)
		np.random.seed(seed)
		torch.manual_seed(seed)
		torch.cuda.manual_seed(seed)
		torch.backends.cudnn.deterministic = True

	def forward(self, h, r, t, h_indices=None, r_indices=None, t_indices=None, slice_size=None, slice_dim=None, model=None):
		h, b_h, _ = h
		t, b_t, _ = t
		r_vec, r_mat = r
		r_ids = self.kge.markers[r_indices]
		embs = model.entity_representations[0](indices=to_cuda(torch.LongTensor(range(self.kge.nfeatures)), self.kge.cuda_on))
		
		## I. For relations != "positive", "negative", replace user/item embeddings by their formula with feature embeddings
		## I.1. get the corresponding batch indices
		h_i_cond = (r_ids.abs()!=1)&(h_indices>=self.kge.nfeatures)&(h_indices<self.kge.nfeatures+self.kge.nitems)
		h_u_cond = (r_ids.abs()!=1)&(h_indices>=(self.kge.nfeatures+self.kge.nitems))&(h_indices<(self.kge.nfeatures+self.kge.nitems+self.kge.nusers))
		t_i_cond = (r_ids.abs()!=1)&(t_indices>=self.kge.nfeatures)&(t_indices<self.kge.nfeatures+self.kge.nitems)
		t_u_cond = (r_ids.abs()!=1)&(t_indices>=(self.kge.nfeatures+self.kge.nitems))&(h_indices<(self.kge.nfeatures+self.kge.nitems+self.kge.nusers))
		h_ids_item = h_indices[h_i_cond]
		h_ids_user = h_indices[h_u_cond]
		t_ids_item = t_indices[t_i_cond]
		t_ids_user = t_indices[t_u_cond]
		
		#print([x.size(dim=0) for x in [h_ids_item,h_ids_user,t_ids_item,t_ids_user, r_ids[r_ids==1], h_indices[(r_ids==1)&(h_indices<self.kge.nfeatures)], t_indices[(r_ids==1)&(t_indices<self.kge.nfeatures)] ]])
		## I.2. get the corresponding feature vectors
		Sh = self.kge.features[h_ids_item-self.kge.nfeatures,:]
		St = self.kge.features[t_ids_item-self.kge.nfeatures,:]
		Ph = self.kge.features[h_ids_user-self.kge.nfeatures,:]
		Pt = self.kge.features[t_ids_user-self.kge.nfeatures,:]
		
		h_ = h.clone()
		t_ = t.clone()
		## I.3. compute the formula with feature embeddings and replace in triplets
		h_[h_i_cond,:] = self.FM.structure(Sh, embs)
		h_[h_u_cond,:] = self.FM.structure(Ph, embs)
		t_[t_i_cond,:] = self.FM.structure(St, embs)
		t_[t_u_cond,:] = self.FM.structure(Pt, embs)

		## II. For relations == "positive", "negative", replace user/item/feature embeddings by their feature vectors this time (input to the FM)
		## II.1. get the corresponding batch indices
		h_g_condPred = (r_ids.abs()==1)&(h_indices<self.kge.nfeatures)
		t_g_condPred = (r_ids.abs()==1)&(t_indices<self.kge.nfeatures)
		h_i_condPred = (r_ids.abs()==1)&(h_indices>=self.kge.nfeatures)&(h_indices<self.kge.nfeatures+self.kge.nitems)
		h_u_condPred = (r_ids.abs()==1)&(h_indices>=(self.kge.nfeatures+self.kge.nitems))&(h_indices<(self.kge.nfeatures+self.kge.nitems+self.kge.nusers))
		t_i_condPred = (r_ids.abs()==1)&(t_indices>=self.kge.nfeatures)&(t_indices<self.kge.nfeatures+self.kge.nitems)
		t_u_condPred = (r_ids.abs()==1)&(t_indices>=(self.kge.nfeatures+self.kge.nitems))&(h_indices<(self.kge.nfeatures+self.kge.nitems+self.kge.nusers))
		hh_ig = h_indices[h_g_condPred]
		tt_ig = t_indices[t_g_condPred]		
		hh_ii = h_indices[h_i_condPred]
		hh_iu = h_indices[h_u_condPred]
		tt_ii = t_indices[t_i_condPred]
		tt_iu = t_indices[t_u_condPred]
		
		#print([x.size(dim=0) for x in [hh_ig, tt_ig, hh_ii, hh_iu, tt_ii, tt_iu, r_ids[r_ids!=1]]])
		## II.2. get the corresponding feature vectors for item and user
		Shh = self.kge.features[hh_ii-self.kge.nfeatures,:]
		Stt = self.kge.features[tt_ii-self.kge.nfeatures,:]
		Phh = self.kge.features[hh_iu-self.kge.nfeatures,:]
		Ptt = self.kge.features[tt_iu-self.kge.nfeatures,:]
		## II.3. for features, the corresponding feature vector is [0, ..., 0] with a 1 at the position of the feature
		if (self.kge.nfeatures<=1000):
			I = to_cuda(torch.eye(self.kge.nfeatures), self.cuda_on)
			Ghh = I[hh_ig,:]
			Gtt = I[tt_ig,:]
		else:
			hh_inds = to_cuda(torch.LongTensor(torch.arange(hh_ig.size(dim=0))), self.cuda_on)			
			Ghh = to_cuda(torch.zeros(hh_ig.size(dim=0),self.kge.nfeatures), self.cuda_on)
			Ghh[hh_inds, hh_ig] = 1
			tt_inds = to_cuda(torch.LongTensor(torch.arange(tt_ig.size(dim=0))), self.cuda_on)	
			Gtt = to_cuda(torch.zeros(tt_ig.size(dim=0),self.kge.nfeatures), self.cuda_on)
			Gtt[tt_inds, tt_ig] = 1
			
		## II.4. replace in triplets by feature vectors
		hh = to_cuda(torch.empty(h.size(dim=0), embs.size(dim=0)), self.kge.cuda_on)
		tt = to_cuda(torch.empty(h.size(dim=0), embs.size(dim=0)), self.kge.cuda_on)
		#hh[:,:] = -1000
		#tt[:,:] = -1000
		#print("*********************")
		hh[h_g_condPred,:] = Ghh
		tt[t_g_condPred,:] = Gtt
		hh[h_i_condPred,:] = Shh
		hh[h_u_condPred,:] = Phh
		tt[t_i_condPred,:] = Stt
		tt[t_u_condPred,:] = Ptt
		
		#print((hh.size(dim=0), hh[hh[:,0]==-1000].size(dim=0)))
		#print((tt.size(dim=0), tt[tt[:,0]==-1000].size(dim=0)))
		
		#print((hh.size(), Shh.size(), Phh.size(), Ghh.size()))
		#print((tt.size(), Stt.size(), Ptt.size(), Gtt.size()))

		#print("-----------------------------------")
		
		#print(("range_pair1", hh.min(), hh.max()))
		#print(("range_pair2", tt.min(), tt.max()))

		## III. run MuRE, resp. the FM on triplets and return the corresponding output vector
		## TODO: implement negative relations as well s(h, r-, t) = s(-h, r+, t) for relations != "positive", "negative"
		rshp = lambda x, ric : torch.reshape(x, ((ric).sum(), h.shape[1] if (len(h.shape)>2) else 1))
		preds = to_cuda(torch.empty(h.shape[0], h.shape[1] if (len(h.shape)>2) else 1), self.kge.cuda_on)
		#for ri in np.unique(r_ids[r_ids.abs()!=1]):
                #    cond = r_ids==ri
                #    preds_notrec = self.__class__.func(h=h_[cond,:], b_h=b_h[cond], r_vec=r_vec[cond,:], r_mat=r_mat[cond,:], t=t_[cond,:], b_t=b_t[cond])
		cond = r_ids.abs()!=1
		preds_notrec = self.__class__.func(h=h_[cond,:], b_h=b_h[cond], r_vec=r_vec[cond,:], r_mat=r_mat[cond,:], t=t_[cond,:], b_t=b_t[cond])
		preds[cond,:] = rshp(preds_notrec, cond)
		preds[r_ids==1,:] = rshp(self.FM((hh[r_ids==1,:], tt[r_ids==1,:], embs)), r_ids==1)
		preds[r_ids==-1,:] = rshp(-self.FM((hh[r_ids==-1,:], tt[r_ids==-1,:], embs)), r_ids==-1)
		
		#print(("range_pred", preds_notrec.min(), preds_notrec.max(), preds_rec.min(), preds_rec.max(), preds.min(),preds.max()))
		#assert not np.isnan(preds.max().detach().numpy())
		
		return preds
		
##############################################################################
## Create a KGE from a Dataset + relation type annotations                  ##
##############################################################################
class KGE(object):
    def __init__(self, dataset, partial_kge=None, cuda_on=False, sim_thres=0, use_ratings=False, positive="positive", negative="negative", kge_name=None):
        super().__init__()
        assert dataset.items.shape[0]==dataset.users.shape[0]
        assert all([i in dataset.user_features for i in dataset.item_features])
        assert all([i in dataset.item_features for i in dataset.user_features])
        self.positive = positive
        self.negative = negative
        if ((kge_name is not None) and os.path.exists(kge_name+".tsv")):
            self.load_triplets(kge_name)
        else:
            dts = deepcopy(dataset)
            dts.item_list = ["I%s" %i for i,_ in enumerate(dts.item_list)]
            dts.user_list = ["U%s" %i for i,_ in enumerate(dts.user_list)]
            dts.item_features = ["F%s" %i for i,_ in enumerate(dts.item_features)]
            ordi = np.argsort(dts.item_list)
            ordu = np.argsort(dts.user_list)
            ordf = np.argsort(dts.item_features)
            dts.users = coo_array(dts.users.toarray()[ordf,:][:,ordu])
            dts.items = coo_array(dts.items.toarray()[ordf,:][:,ordi])
            dts.item_list = np.array(dts.item_list)[ordi].tolist()
            dts.user_list = np.array(dts.user_list)[ordu].tolist()
            dts.item_features = np.array(dts.item_features)[ordf].tolist()
            dts.user_features = dts.item_features
            ## Create KGE from dataset
            triplets = []
            ## Connect items and users using the association matrix
            ROW = np.array(dts.item_list)[dts.ratings.row]
            COL = np.array(dts.user_list)[dts.ratings.col]
            npos = (dts.ratings.data>0).sum()
            if (npos>0):
                pos_nb = np.tile(np.array(positive), (npos, 1))
                pos_it = ROW[dts.ratings.data>0].reshape(-1, 1)
                pos_us = COL[dts.ratings.data>0].reshape(-1, 1)
                triplets += [np.hstack((pos_it, pos_nb, pos_us))]
            nneg = (dts.ratings.data<0).sum()
            if (nneg>0):
                neg_nb = np.tile(np.array(negative), (nneg, 1))
                neg_it = ROW[dts.ratings.data<0].reshape(-1, 1)
                neg_us = COL[dts.ratings.data<0].reshape(-1, 1    )
                triplets += [np.hstack((neg_it, neg_nb, neg_us))]
            if (sim_thres>=0):
                ## Connect items using the item feature matrix + association matrix
                ITEMS = np.hstack((dts.items.toarray().T, dts.ratings.toarray())) if (use_ratings) else dts.items.toarray().T
                cosmat = pairwise_distances(ITEMS, metric="cosine")
                cosmat[cosmat<sim_thres] = 0
                cosmat = coo_array(cosmat)
                nsimi = (cosmat>0).sum()
                if (nsimi>0):
                    nsimi_nb = np.tile(np.array("item-sim-pos"), (nsimi, 1))
                    nsimi_i1 = np.array(dts.item_list)[cosmat.row].reshape(-1, 1)
                    nsimi_i2 = np.array(dts.item_list)[cosmat.col].reshape(-1, 1)
                    triplets += [np.hstack((nsimi_i1, nsimi_nb, nsimi_i2))]
                ## Connect users using the user feature matrix + association matrix
                USERS = np.hstack((dts.users.toarray().T, dts.ratings.toarray().T)) if (use_ratings) else dts.users.toarray().T
                cosmat = pairwise_distances(USERS, metric="cosine")
                cosmat[cosmat<sim_thres] = 0
                cosmat = coo_array(cosmat)
                nsimu = (cosmat>0).sum()
                if (nsimu>0):
                    nsimu_nb = np.tile(np.array("user-sim-pos"), (nsimu, 1))
                    nsimu_u1 = np.array(dts.user_list)[cosmat.row].reshape(-1, 1)
                    nsimu_u2 = np.array(dts.user_list)[cosmat.col].reshape(-1, 1)
                    triplets += [np.hstack((nsimu_u1, nsimu_nb, nsimu_u2))]
            ## Connect items and features using the item feature matrix
            ROW = np.array(dts.item_features)[dts.items.row]
            COL = np.array(dts.item_list)[dts.items.col]
            nfeati = (dts.items.data>0).sum()
            if (nfeati>0):
                nfeati_nb = np.tile(np.array("item-feature-pos"), (nfeati, 1))
                nfeati_f = ROW[dts.items.data>0].reshape(-1, 1)
                nfeati_it = COL[dts.items.data>0].reshape(-1, 1)
                triplets += [np.hstack((nfeati_f, nfeati_nb, nfeati_it))]
            nfeati = (dts.items.data<0).sum()
            if (nfeati>0):
                nfeati_nb = np.tile(np.array("item-feature-neg"), (nfeati, 1))
                nfeati_f = ROW[dts.items.data<0].reshape(-1, 1)
                nfeati_it = COL[dts.items.data<0].reshape(-1, 1)
                triplets += [np.hstack((nfeati_f, nfeati_nb, nfeati_it))]
            ## Connect users and features using the user feature matrix
            ROW = np.array(dts.user_features)[dts.users.row]
            COL = np.array(dts.user_list)[dts.users.col]
            nfeati = (dts.users.data>0).sum()
            if (nfeati>0):
                nfeati_nb = np.tile(np.array("user-feature-pos"), (nfeati, 1))
                nfeati_f = ROW[dts.users.data>0].reshape(-1, 1)
                nfeati_it = COL[dts.users.data>0].reshape(-1, 1)
                triplets += [np.hstack((nfeati_f, nfeati_nb, nfeati_it))]
            nfeati = (dts.users.data<0).sum()
            if (nfeati>0):
                nfeati_nb = np.tile(np.array("user-feature-neg"), (nfeati, 1))
                nfeati_f = ROW[dts.users.data<0].reshape(-1, 1)
                nfeati_it = COL[dts.users.data<0].reshape(-1, 1)
                triplets += [np.hstack((nfeati_f, nfeati_nb, nfeati_it))]
            self.triplets = np.vstack(tuple(triplets))
            if (partial_kge is not None):
                assert partial_kge.shape[1]==3
                ## non feature, non user, and non item nodes are latent variables
                assert all(np.vectorize(lambda x : x not in [positive, negative])(partial_kge[:,1]))
                if (not all(np.vectorize(lambda x : x in dataset.item_list+dataset.user_list+dataset.item_features+dataset.user_features)(np.unique(partial_kge[:,[0,2]].flatten())))):
                    ppartial_kge = deepcopy(partial_kge)
                    
                    #arr = np.unique(np.array(dataset.item_list+dataset.user_list+dataset.item_features+dataset.user_features))
                    #cond = np.vectorize(lambda x : x in arr)(np.unique(ppartial_kge[:,[0,2]].flatten()))
                    #print(np.unique(ppartial_kge[:,[0,2]].flatten())[cond].shape)
                    
                    ppartial_kge[:,0] = np.vectorize(lambda x : "Z"+x)(ppartial_kge[:,0])
                    ppartial_kge[:,2] = np.vectorize(lambda x : "Z"+x)(ppartial_kge[:,2])
                    
                    #cond = np.vectorize(lambda x : x[1:] in arr)(np.unique(ppartial_kge[:,[0,2]].flatten()))
                    #print(np.unique(ppartial_kge[:,[0,2]].flatten())[cond].shape)

                    M = max(dataset.nitems, max(dataset.nusers, dataset.nitem_features))
                    def replace_i(i_lst):
                        for i in tqdm(i_lst):
                            if (i<dataset.nitems):
                                ppartial_kge[ppartial_kge[:,0]==("Z"+dataset.item_list[i]),0] = "I%d" % i
                                ppartial_kge[ppartial_kge[:,2]==("Z"+dataset.item_list[i]),2] = "I%d" % i
                            if (i<dataset.nusers):
                                ppartial_kge[ppartial_kge[:,0]==("Z"+dataset.user_list[i]),0] = "U%d" % i
                                ppartial_kge[ppartial_kge[:,2]==("Z"+dataset.user_list[i]),2] = "U%d" % i
                            if (i<dataset.nitem_features):
                                ppartial_kge[ppartial_kge[:,0]==("Z"+dataset.item_features[i]),0] = "F%d" % i
                                ppartial_kge[ppartial_kge[:,2]==("Z"+dataset.item_features[i]),2] = "F%d" % i
                        return None
                    replace_i(range(M))
                    self.triplets = np.vstack((self.triplets, ppartial_kge))
                else:
                    self.triplets = np.vstack((self.triplets, partial_kge))
            if (kge_name is not None):
                self.save_triplets(kge_name)
        all_rels = np.unique(self.triplets[:,1].flatten())
        markers_di = {positive: 1, negative: -1}
        markers_di.update({k: (-1)**int("neg" in k)*2 for k in ["item-sim-pos","item-sim-neg"] if (k in all_rels)})
        markers_di.update({k: (-1)**int("neg" in k)*3 for k in ["user-sim-pos","user-sim-neg"] if (k in all_rels)})
        markers_di.update({k: (-1)**int("neg" in k)*4 for k in ["item-feature-pos","item-feature-neg"] if (k in all_rels)})
        markers_di.update({k: (-1)**int("neg" in k)*5 for k in ["user-feature-pos","user-feature-neg"] if (k in all_rels)})
        self.markers = to_cuda(torch.LongTensor([markers_di.get(k, 0) for k in sorted(set(self.triplets[:,1].flatten().tolist()))]), cuda_on)
        self.nfeatures = dataset.nitem_features
        self.nusers = dataset.nusers
        self.nitems = dataset.nitems
        features = np.vstack((normalization(dataset.items.toarray().T), normalization(dataset.users.toarray().T)))
        self.features = to_cuda(torch.Tensor(features), cuda_on)
        self.cuda_on = cuda_on
        
    def save_triplets(self, kge_name=None):
        assert kge_name is not None
        pd.DataFrame(self.triplets, index=range(self.triplets.shape[0]), columns=range(self.triplets.shape[1])).to_csv(kge_name+".tsv", sep="\t", header=False, index=False)

    def load_triplets(self, kge_name=None):
        assert kge_name is not None
        assert os.path.exists(kge_name+".tsv")
        triplets = pd.read_csv(kge_name+".tsv", sep="\t", header=None).values
        assert triplets.shape[1]==3
        assert any(np.vectorize(lambda x : x in [self.positive, self.negative])(triplets[:,1]))
        #assert any(np.vectorize(lambda x : x in dataset.item_list+dataset.user_list+dataset.item_features+dataset.user_features)(np.unique(triplets[:,[0,2]].flatten())))
        self.triplets = triplets
        
    def print(self):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8,8))
        edges = list(map(tuple, self.triplets[:,[0,2,1]]))
        g = ig.Graph.TupleList(edges, directed=False, edge_attrs="name")
        pal_ent = {'U': "orange", 'I': "gray", 'F': "purple"}
        for v in g.vs:
        	v.update_attributes({"color": pal_ent[v.attributes()["name"][0]] })
        cm = plt.cm.get_cmap("tab20b").colors
       	pal_rel = {r: cm[(ir+7)%len(cm)] for ir, r in enumerate(np.unique(self.triplets[:,1].flatten())) if (r not in [self.positive, self.negative])}
        for e in g.es:
        	e.update_attributes({"color": "red" if (e.attributes()["name"]==self.negative) else ("green" if (e.attributes()["name"]==self.positive) else pal_rel[e.attributes()["name"]]) })
        ig.plot(
            g,
            target=ax,
            vertex_size=20,
            edge_width=0.7
        )
        handles = [Line2D([0], [0], label=k, color=v) for (k,v) in pal_rel.items()]
        handles += [Line2D([0], [0], label=k, color={self.positive: "green", self.negative:"red"}[k]) for k in [self.positive, self.negative]]
        handles += [Line2D([0], [0], label=v, marker='o', markersize=10, markeredgecolor='k', markerfacecolor=pal_ent[v[0].upper()], linestyle='') for v in ["user", "item", "feature"]]
        ax.legend(handles=handles, loc='upper right')
        plt.show()

######################################################################################################
## Modifying the Model class in PyKeen to get access to triplet indices at run time                 ##
######################################################################################################
#from pykeen.regularizers import LpRegularizer, OrthogonalityRegularizer
def make_model_cls(dimensions,interaction_instance, entity_representations_kwargs=None, relation_representations_kwargs=None):
	entity_representations_kwargs, relation_representations_kwargs = _normalize_representation_kwargs(
		dimensions=dimensions, interaction=interaction_instance.__class__,  # type: ignore
		entity_representations_kwargs=entity_representations_kwargs, relation_representations_kwargs=relation_representations_kwargs
	)
	class JELIModel(ERModel):
		def __init__(self, **kwargs):
			super().__init__(interaction=interaction_instance, entity_representations_kwargs=entity_representations_kwargs, relation_representations_kwargs=relation_representations_kwargs, **kwargs)
			## regularizer
			#regularizer = LpRegularizer(**dict(
			#	weight=10, #0.001 / 2,
			#	p=1.0, #2.0,
			#	normalize=True,
			#	apply_only_once=True,
			#)) #None
			#regularizer = OrthogonalityRegularizer(weight=10)
			#print([x for x in self.interaction.parameters()])
			#if regularizer is not None:
			#	self.append_weight_regularizer(parameter=self.interaction.parameters(), regularizer=regularizer,
			#)
			#print([x for x in regularizer.parameters()])
			#raise ValueError

		def forward(self,h_indices,r_indices,t_indices,slice_size=None,slice_dim=0,*,mode=None):
			if not self.entity_representations or not self.relation_representations:
				raise NotImplementedError("repeat scores not implemented for general case.")
			h, r, t = self._get_representations(h=h_indices, r=r_indices, t=t_indices, mode=mode)
			return self.interaction.forward(h=h, r=r, t=t, h_indices=h_indices, r_indices=r_indices, t_indices=t_indices, slice_size=slice_size, slice_dim=slice_dim, model=self)

		def score_hrt(self, hrt_batch, *, mode=None):
			h, r, t = self._get_representations(h=hrt_batch[:, 0], r=hrt_batch[:, 1], t=hrt_batch[:, 2], mode=mode)
			return self.interaction.forward(h=h, r=r, t=t, h_indices=hrt_batch[:, 0], r_indices=hrt_batch[:, 1], t_indices=hrt_batch[:, 1], model=self)

		def score_t(self,hr_batch,*,slice_size=None,mode=None,tails=None):
			self._check_slicing(slice_size=slice_size)
			# add broadcast dimension
			hr_batch = hr_batch.unsqueeze(dim=1)
			h, r, t = self._get_representations(h=hr_batch[..., 0], r=hr_batch[..., 1], t=tails, mode=mode)
			# unsqueeze if necessary
			if tails is None or tails.ndimension() == 1:
				t = parallel_unsqueeze(t, dim=0)
			return repeat_if_necessary(scores=self.interaction.forward(h=h, r=r, t=t, h_indices=hr_batch[..., 0], r_indices=hr_batch[..., 1], t_indices=tails, slice_size=slice_size, slice_dim=1, model=self), representations=self.entity_representations, num=self._get_entity_len(mode=mode) if tails is None else tails.shape[-1])

		def score_r(self,ht_batch,*,slice_size=None,mode=None,relations=None):  # noqa: D102
			self._check_slicing(slice_size=slice_size)
			# add broadcast dimension
			ht_batch = ht_batch.unsqueeze(dim=1)
			h, r, t = self._get_representations(h=ht_batch[..., 0], r=relations, t=ht_batch[..., 1], mode=mode)
			# unsqueeze if necessary
			if relations is None or relations.ndimension() == 1:
				r = parallel_unsqueeze(r, dim=0)
			return repeat_if_necessary(scores=self.interaction.forward(h=h, r=r, t=t, h_indices=ht_batch[..., 0], r_indices=relations, t_indices=rt_batch[..., 1], slice_size=slice_size, slice_dim=1, model=self), representations=self.relation_representations, num=self.num_relations if relations is None else relations.shape[-1])

		def score_h(self,rt_batch,*,slice_size=None,mode=None,heads=None):  # noqa: D102
			self._check_slicing(slice_size=slice_size)
			# add broadcast dimension
			rt_batch = rt_batch.unsqueeze(dim=1)
			h, r, t = self._get_representations(h=heads, r=rt_batch[..., 0], t=rt_batch[..., 1], mode=mode)
			# unsqueeze if necessary
			if heads is None or heads.ndimension() == 1:
				h = parallel_unsqueeze(h, dim=0)
			return repeat_if_necessary(scores=self.interaction.forward(h=h, r=r, t=t, h_indices=heads, r_indices=rt_batch[..., 0], t_indices=rt_batch[..., 1], slice_size=slice_size, slice_dim=1, model=self),representations=self.entity_representations,num=self._get_entity_len(mode=mode) if heads is None else heads.shape[-1])

	JELIModel._interaction = interaction_instance
	return JELIModel
	
#####################################################################
## Negative sampler: selecting hard examples to discriminate       ##
#####################################################################
from pykeen.sampling import PseudoTypedNegativeSampler
from pykeen.triples import CoreTriplesFactory, TriplesFactory

## https://pykeen.readthedocs.io/en/stable/api/pykeen.sampling.PseudoTypedNegativeSampler.html
class TypedNegativeSampler(PseudoTypedNegativeSampler):

    def __init__(self, *, mapped_triples: CoreTriplesFactory, cuda_on=False, **kwargs):
        super().__init__(mapped_triples=mapped_triples, **kwargs)
        self.cuda_on = cuda_on
        return
    
    def corrupt_batch(self, positive_batch: torch.LongTensor):  # noqa: D102
        batch_size = positive_batch.shape[0]
        
        # shape: (batch_size, num_neg_per_pos, 3)
        negative_batch = positive_batch.unsqueeze(dim=1).repeat(1, self.num_negs_per_pos, 1)

        # Uniformly sample from head/tail offsets
        r = positive_batch[:, 1].long()                             ## all relations in positive batches
        start_heads = self.offsets[2 * r].unsqueeze(dim=-1)         ## heads for relation r
        start_tails = self.offsets[2 * r + 1].unsqueeze(dim=-1)     ## tails for relation r
        end = self.offsets[2 * r + 2].unsqueeze(dim=-1)             
        num_choices = end - start_heads
        negative_ids = start_heads + (torch.rand(size=(batch_size, self.num_negs_per_pos)) * num_choices).long()

        # get corresponding entity
        entity_id = self.data[negative_ids]

        # and position within triple (0: head, 2: tail)
        triple_position = 2 * (negative_ids >= start_tails).long()

        # write into negative batch
        negative_batch[
            torch.arange(batch_size, device=negative_batch.device).unsqueeze(dim=-1),
            torch.arange(self.num_negs_per_pos, device=negative_batch.device).unsqueeze(dim=0),
            triple_position,
        ] = entity_id.type(negative_batch.dtype)

        return negative_batch
 
######################################################
## Simplifying the PyKeen pipeline                  ##
######################################################       
def pipeline(kge=None,tf=None,model=None,model_kwargs=None,dimensions=None,loss=None,loss_kwargs=None,optimizer=None,optimizer_kwargs=None,clear_optimizer=True,lr_scheduler=None,lr_scheduler_kwargs=None,training_loop=None,training_loop_kwargs=None,cuda_on=False,negative_sampler=None,negative_sampler_kwargs=None,epochs=None,training_kwargs=None,metadata={},device=None,random_seed=None):
    assert (kge is not None) or (tf is not None)
    assert not ((kge is not None) and (tf is not None))
    if (tf is None):
        tf = TriplesFactory.from_labeled_triples(kge.triplets)
    if training_kwargs is None:
        training_kwargs = {}
    training_kwargs = dict(training_kwargs)
    _random_seed = random_seed
    set_random_seed(_random_seed)
    _device: torch.device = resolve_device(device)
    model_instance, model_kwargs = _build_model_helper(model=model, model_kwargs=model_kwargs, loss=loss,
        loss_kwargs=loss_kwargs, _device=_device, regularizer=None, regularizer_kwargs=None, _random_seed=_random_seed, training_triples_factory=tf)
    model_instance = model_instance.to(_device)
    optimizer_kwargs = dict(optimizer_kwargs or {})
    optimizer_instance = optimizer_resolver.make(optimizer, optimizer_kwargs, params=model_instance.get_grad_params())
    for key, value in optimizer_instance.defaults.items():
        optimizer_kwargs.setdefault(key, value)
    if lr_scheduler is None:
        lr_scheduler_instance = None
    else:
        lr_scheduler_instance = lr_scheduler_resolver.make(
            lr_scheduler,
            lr_scheduler_kwargs,
            optimizer=optimizer_instance,
        )
    training_loop_cls = training_loop_resolver.lookup(training_loop)
    if training_loop_kwargs is None:
        training_loop_kwargs = {}
    training_loop_kwargs = dict(training_loop_kwargs)
    if (negative_sampler=="PseudoTypedNegativeSampler"):
        negative_sampler = TypedNegativeSampler(mapped_triples=tf.mapped_triples, cuda_on=cuda_on, **negative_sampler_kwargs)
        training_loop_instance = training_loop_cls(model=model_instance, triples_factory=tf, 
             negative_sampler=negative_sampler, optimizer=optimizer_instance, lr_scheduler=lr_scheduler_instance, **training_loop_kwargs)
    elif (negative_sampler is not None) and (not issubclass(training_loop_cls, SLCWATrainingLoop)):
        raise ValueError("Can not specify negative sampler with LCWA")
    else:
        if (negative_sampler is None):
            negative_sampler_cls = None
        negative_sampler_cls = negative_sampler_resolver.lookup(negative_sampler)
        training_loop_kwargs.update(negative_sampler=negative_sampler_cls, negative_sampler_kwargs=negative_sampler_kwargs)
        training_loop_instance = training_loop_cls(model=model_instance, triples_factory=tf, 
               optimizer=optimizer_instance, lr_scheduler=lr_scheduler_instance, **training_loop_kwargs)
    if "stopper" in training_kwargs:
        stopper = training_kwargs.pop("stopper")
    else:
        stopper = None
    stopper_kwargs = dict({})
    stopper_instance = stopper_resolver.make(stopper, model=model_instance, training_triples_factory=tf, **stopper_kwargs)
    if epochs is not None:
        training_kwargs["num_epochs"] = epochs
    training_kwargs["use_tqdm"] = True
    assert "num_epochs" in training_kwargs 
    assert "batch_size" in training_kwargs
    with torch.autograd.set_detect_anomaly(True):
        losses = training_loop_instance.train(triples_factory=tf, stopper=stopper_instance, clear_optimizer=clear_optimizer, **training_kwargs)
    assert losses is not None  # losses is only none if it's doing search mode
    return dict(
        random_seed=_random_seed,
        model=model_instance,
        training=tf, #training,
        training_loop=training_loop_instance,
        losses=losses,
        stopper=stopper_instance,
        metadata=metadata,
    )

######################################################
## Utils                                            ##
######################################################
def broadcast_cat(tensors, dim):
    if len(tensors) == 0:
        raise ValueError("Must pass at least one tensor.")
    if len({x.ndimension() for x in tensors}) != 1:
        raise ValueError(
            f"The number of dimensions has to be the same for all tensors, but is {set(t.shape for t in tensors)}",
        )
    if len(tensors) == 1:
        return tensors[0]
    if dim < 0:
        dim = tensors[0].ndimension() + dim
    repeats = [
        [1 for _ in t.shape]
        for t in tensors
    ]
    for i, dims in enumerate(zip(*(t.shape for t in tensors))):
        if i == dim:
            continue
        d_max = max(dims)
        if not {1, d_max}.issuperset(dims):
            raise ValueError(f"Tensors have invalid shape along {i} dimension: {set(dims)}")
        for j, td in enumerate(dims):
            if td != d_max:
                repeats[j][i] = d_max
    tensors = [
        t.repeat(*r)
        for t, r in zip(tensors, repeats)
    ]
    return torch.cat(tensors, dim=dim)

def parallel_unsqueeze(x, dim):
    xs = upgrade_to_sequence(x)
    xs = [xx.unsqueeze(dim=dim) for xx in xs]
    return xs[0] if len(xs) == 1 else xs
    
def _normalize_representation_kwargs(dimensions, interaction, entity_representations_kwargs, relation_representations_kwargs):
    if isinstance(dimensions, int):
        dimensions = {"d": dimensions}
    assert isinstance(dimensions, dict)
    if set(dimensions) < interaction.get_dimensions():
        raise ValueError
    if entity_representations_kwargs is None:
        entity_representations_kwargs = [
            dict(shape=tuple(dimensions[d] for d in shape)) for shape in interaction.entity_shape
        ]
    elif not isinstance(entity_representations_kwargs, Sequence):
        entity_representations_kwargs = [entity_representations_kwargs]
    if relation_representations_kwargs is None:
        relation_representations_kwargs = [
            dict(shape=tuple(dimensions[d] for d in shape)) for shape in interaction.relation_shape
        ]
    elif not isinstance(relation_representations_kwargs, Sequence):
        relation_representations_kwargs = [relation_representations_kwargs]
    return entity_representations_kwargs, relation_representations_kwargs
    
def repeat_if_necessary(scores, representations, num):
    if representations:
        return scores
    return scores.repeat(1, num)
    
def normalization_threshold(x, t=1e-3): ## n x F (threshold)
    ## Quantile normalization and [0,1]-normalization and sparsity
    vals_norm = normalize(quantile_normalize(np.nan_to_num(x, nan=0)), axis=1, norm='l1')
    return np.multiply(vals_norm, (np.abs(vals_norm)>=t).astype(int))
    
def normalization(x, _q=0.9): ## n x F (keep 100q^th-extreme values)
    ## Quantile normalization and [0,1]-normalization and sparsity
    vals_norm = normalize(quantile_normalize(np.nan_to_num(x, nan=0)), axis=1, norm='l1')
    t, _t = np.quantile(vals_norm, _q/2), np.quantile(vals_norm, (1-_q)/2)
    return np.multiply(vals_norm, ((vals_norm<=t)|(vals_norm>=_t)).astype(int))

def to_cuda(x,cuda_on):
    return x.cuda() if (cuda_on) else x
