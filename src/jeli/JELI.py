#coding: utf-8

## JELI: "Joint Embedding Learning for Interpretability"
from stanscofi.models import BasicModel
from stanscofi.preprocessing import CustomScaler
from stanscofi.datasets import Dataset
from scipy.sparse import coo_array

from .JELIImplementation import pipeline, to_cuda, make_model_cls, MuRE_RHOFM, KGE, normalization

import torch
import pickle
import numpy as np
import pandas as pd

class JELI(BasicModel):
    def __init__(self, params=None):
        params_ = self.default_parameters()
        if (params is not None):
            params_.update(params)
        super(JELI, self).__init__(params_)
        self.scalerS, self.scalerP = None, None
        self.name = "JELI" 
        self.model = None
        self.features = None
        self.nitems = None
        self.feature_list = None

    def default_parameters(self):
        cuda_on = False
        params = {
            "epochs": 25,
            "batch_size": 1032, "lr": 0.001, 
            "n_dimensions": 50,
            "optimizer": "Adam", "loss": "SoftMarginRankingLoss",
            "loss_kwargs": None, 
            #"regularizer": None, "regularizer_kwargs": None,
            "negative_sampler": "PseudoTypedNegativeSampler",
            "negative_sampler_kwargs": {}, 
            #"negative_sampler": "basic",
            #"negative_sampler_kwargs": dict(corruption_scheme=['head']), 
            #"negative_sampler_kwargs": dict(filtered=True, filterer='bloom', num_negs_per_pos=3, filterer_kwargs=dict(error_rate=0.0001)),
            "cuda_on": cuda_on, 
            "p_norm": 2, ## norm of the MuRE interaction model
            "order": 2, ## order of the HOFM
            "structure": "linear", ## "linear" or function lambda x, w: torch.matmul(x, w)
            "sim_thres": -1, "use_ratings": False,
            "random_seed": 1234,
            "partial_kge": None,
            "frozen": False,
            "kge_name": None,
        }
        return params

    def preprocessing(self, dataset, is_training=True, inf=2):
        if (is_training):
            ## Merge user and item sets to get the full feature set
            S = pd.DataFrame(dataset.items.toarray(), index=dataset.item_features, columns=["i."+i for i in dataset.item_list])
            P = pd.DataFrame(dataset.users.toarray(), index=[f if ((f in dataset.item_features) and (dataset.nitem_features==dataset.nuser_features)) else ("u."+f if (f in dataset.item_features) else f) for f in dataset.user_features], columns=["u."+u for u in dataset.user_list])
            SP = S.join(P, how="outer").fillna(0)
            self.feature_list = [dataset.item_features, dataset.user_features, list(SP.index)]
            if (self.scalerS is None):
                self.scalerS = CustomScaler(posinf=inf, neginf=-inf)
            S_ = self.scalerS.fit_transform(SP.values[:,:dataset.nitems], subset=None)
            if (self.scalerP is None):
                self.scalerP = CustomScaler(posinf=inf, neginf=-inf)
            P_ = self.scalerP.fit_transform(SP.values[:,dataset.nitems:], subset=None)
            proc_dataset = Dataset(
                ratings=pd.DataFrame(dataset.ratings.toarray(), index=dataset.item_list, columns=dataset.user_list), 
                users=pd.DataFrame(P_, index=self.feature_list[-1], columns=dataset.user_list), 
                items=pd.DataFrame(S_, index=self.feature_list[-1], columns=dataset.item_list)
            )
            proc_dataset.folds = dataset.folds
            kge = KGE(proc_dataset, partial_kge=self.partial_kge, cuda_on=self.cuda_on, sim_thres=self.sim_thres, use_ratings=self.use_ratings, kge_name=self.kge_name)
            self.features = kge.features
            self.nitems = kge.nitems
            return [kge]
        else:
            assert self.features is not None
            assert self.nitems is not None
            assert self.feature_list is not None
            items, users = self.features[dataset.folds.row,:].T, self.features[self.nitems+dataset.folds.col,:].T
            return [items, users]
        
    def model_fit(self, kge):
        results = pipeline(kge, model=make_model_cls(dimensions={"":1, "d": self.n_dimensions}, interaction_instance=MuRE_RHOFM(self.n_dimensions, self.order, self.structure, kge, frozen=self.frozen, p_norm=self.p_norm, cuda_on=self.cuda_on, random_seed=self.random_seed)),
            optimizer=self.optimizer, optimizer_kwargs={'lr': self.lr}, training_kwargs={'batch_size': self.batch_size, "stopper": None},
            training_loop='sLCWA', 
            model_kwargs=None, 
            #model_kwargs=dict(regularizer=self.regularizer, regularizer_kwargs=self.regularizer_kwargs),
            negative_sampler=self.negative_sampler, cuda_on=self.cuda_on,
            negative_sampler_kwargs=self.negative_sampler_kwargs, 
            loss=self.loss, loss_kwargs=self.loss_kwargs, epochs=self.epochs, random_seed=self.random_seed)
        feature_embeddings = results["model"].entity_representations[0](indices=to_cuda(torch.LongTensor(range(kge.nfeatures)), kge.cuda_on)).detach().cpu()
        self.model = {
		"feature_embeddings": feature_embeddings,
		"features": self.features,
		"feature_list": self.feature_list,
		"nitems": self.nitems,
		"model": results["model"].interaction,
		"losses": results["losses"],  ##
		"random_seed": results["random_seed"],
        }
	
    def save(self, fname="model.pkl"):
        assert self.model is not None
        torch.save(self.model, fname, pickle_protocol=pickle.HIGHEST_PROTOCOL)
        
    def load(self, fname="model.pkl"):
        self.model = torch.load(fname, map_location=torch.device('cpu'))
        self.features = self.model["features"]
        self.nitems = self.model["nitems"]

    def model_predict_proba(self, item, user): ### already normalized, see preprocessing
        assert self.model is not None
        iitem, uuser = to_cuda(torch.Tensor(item).T, self.cuda_on), to_cuda(torch.Tensor(user).T, self.cuda_on)
        scores = torch.sigmoid(self.model["model"].FM((iitem, uuser, to_cuda(self.model["feature_embeddings"], self.cuda_on)))).detach().cpu().numpy().flatten()
        return scores
        
    def transform(self, entity, is_item=False): ## entity of size F x n
        assert self.model is not None
        assert self.feature_list is not None
        if (is_item):
            entity_ = pd.DataFrame(1, index=self.feature_list[-1], columns=["ones"]).join(entity, how="outer").values[:,1:].T
        else:
            entity__ = pd.DataFrame(1, index=self.feature_list[1], columns=["ones"]).join(entity, how="outer")[entity.columns]
            nitems, nusers = len(self.feature_list[0]), len(self.feature_list[1])
            entity__.index = [f if ((f in self.feature_list[0]) and (nitems==nusers)) else ("u."+f if (f in self.feature_list[0]) else f) for f in self.feature_list[1]]
            entity_ = pd.DataFrame(1, index=self.feature_list[-1], columns=["ones"]).join(entity__, how="outer").values[:,1:].T
        entity_ = to_cuda(torch.Tensor(normalization(entity_)), self.cuda_on)
        return self.model["model"].FM.structure(entity_, to_cuda(self.model["feature_embeddings"], self.cuda_on)).detach().cpu()
        
    def score(self, item, user):
        assert self.model is not None
        assert self.feature_list is not None
        item__ = pd.DataFrame(1, index=self.feature_list[-1], columns=["ones"]).join(item, how="outer").values[:,1:].T
        user__ = pd.DataFrame(1, index=self.feature_list[1], columns=["ones"]).join(user, how="outer")[user.columns]
        nitems, nusers = len(self.feature_list[0]), len(self.feature_list[1])
        user__.index = [f if ((f in self.feature_list[0]) and (nitems==nusers)) else ("u."+f if (f in self.feature_list[0]) else f) for f in self.feature_list[1]]
        user__ = pd.DataFrame(1, index=self.feature_list[-1], columns=["ones"]).join(user__, how="outer").values[:,1:].T
        item_, user_ = [normalization(x) for x in [item__, user__]]
        return self.model_predict_proba(item_.T, user_.T)
