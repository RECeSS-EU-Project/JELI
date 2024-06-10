import unittest
import gc
import torch
import pandas as pd
import numpy as np
from stanscofi.datasets import generate_dummy_dataset, Dataset
from stanscofi.utils import load_dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.validation import AUC, NDCGk, compute_metrics

#from sklearn.decomposition import PCA
#import matplotlib.pyplot as plt

import sys
sys.path.insert(0,"../src/")

from jeli.JELI import JELI

class TestModelClass(unittest.TestCase):

	def test_dd_pipeline(self):
		cuda_on = (torch._C._cuda_getDeviceCount()>0)
		dataset_name = "Synthetic" #"Gottlieb"
		NDIM=50
		folder="./"
		SEED=1234
		
		if (dataset_name == "Synthetic"):
			npositive, nnegative, nfeatures, mean, std = 200, 200, 100, 2, 0.01
			data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=SEED)
			data_args.setdefault("name", "Synthetic")
		else:
			data_args = load_dataset(dataset_name, folder+"datasets/")
		dataset = Dataset(**data_args)

		(train_folds, test_folds), _ = random_simple_split(dataset, 0.2, random_state=1234)
		train = dataset.subset(train_folds)
		test = dataset.subset(test_folds)

		print("Total #features = %d, #ratings = %d" % (train.nitem_features+train.nuser_features, len(train.folds.data)))

		#https://pykeen.readthedocs.io/en/stable/reference/losses.html
		model = JELI({"cuda_on": cuda_on, "n_dimensions": NDIM, "frozen": True, "random_state": SEED, "epochs": 3})

		#_, _, kge = model.preprocessing(train, is_training=True)
		#kge.print()

		model.fit(train)
		scores = model.predict_proba(test)
		model.print_scores(scores)
		predictions = model.predict(scores, threshold=0.5)
		model.print_classification(predictions)

		metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
		print(metrics)
		y_test = (test.folds.toarray()*test.ratings.toarray()).ravel()
		y_test[y_test<1] = 0
		print("(global) AUC = %.3f" % AUC(y_test, scores.toarray().ravel(), 1, 1))
		print("(global) NDCG@%d = %.3f" % (test.nitems, NDCGk(y_test, scores.toarray().ravel(), test.nitems, 1)))

		#test.visualize()
		#test.visualize(predictions=predictions)#, dimred_args={"n_neighbors": 5})

		## Interpretability
		print("")

		print("theta [0]")
		w0 = model.model["model"].FM.theta0
		print(pd.DataFrame(w0.detach().cpu().numpy(), index=[0], columns=["theta0"]))

		print("")

		print("theta1 [features]")
		w1 = model.model["model"].FM.theta1
		print(pd.DataFrame(w1.detach().cpu().numpy(), index=list(range(model.n_dimensions)), columns=["theta1"]))

		print("")

		print("theta [2,...,%d]" % model.order)
		w2 = model.model["model"].FM.theta2
		print(pd.DataFrame(w2.detach().cpu().numpy(), index=list(range(2,model.order+1)), columns=["theta2"]))

		for iid, uid in np.argwhere(train.ratings.toarray()!=0):
			#item, user = [ pd.DataFrame(x,index=f,columns=["%d0" % ii]) for ii, [x,f] in enumerate([ [dataset.items.toarray()[:,[iid]], model.feature_list[0]] , [dataset.users.toarray()[:,[uid]], model.feature_list[1]] ]) ]
			item, user = [ pd.DataFrame(x,index=f,columns=["0"]) for x,f in [ [dataset.items.toarray()[:,[iid]], dataset.item_features] , [dataset.users.toarray()[:,[uid]], dataset.user_features] ] ]
			e1 = model.transform(item, is_item=True) 
			print(e1.size())
			e2 = model.transform(user, is_item=False) 
			print(e2.size())
			vpred = model.score(item, user)
			#item_user = item.join(user, how="outer")
			#item_user.index = model.feature_list[-1]
			#item, user = item_user[[item_user.columns[0]]], item_user[[item_user.columns[1]]]
			#vpred = model.model_predict_proba(item.values,user.values) 
			vtrue = dataset.ratings.toarray()[iid,uid]
			if (vpred*vtrue<0):
				print((float(vpred), vtrue))
			break

		[i1, u1], [i2, u2], [i3, u3] = np.argwhere(train.ratings.toarray()!=0).tolist()[:3]
		#items, users = [ pd.DataFrame(x,index=f,columns=["%d0" % ii, "%d1" % ii, "%d2" % ii]) for ii, [x,f] in enumerate([ [dataset.items.toarray()[:,[i1,i2,i3]], model.feature_list[0]] , [dataset.users.toarray()[:,[u1, u2, u3]], model.feature_list[1]] ]) ]
		items, users = [ pd.DataFrame(x,index=f,columns=["0","1","2"]) for x,f in [ [dataset.items.toarray()[:,[i1,i2,i3]], dataset.item_features] , [dataset.users.toarray()[:,[u1, u2, u3]], dataset.user_features] ] ]
		e1 = model.transform(items, is_item=True) 
		print(e1.size())
		e2 = model.transform(users, is_item=False) 
		print(e2.size())
		#items_users = item.join(user, how="outer")
		#items_users.index = model.feature_list[-1]
		#items, users = items_users[items_users.columns[:3]], items_users[items_users.columns[3:]]
		#vpred = model.model_predict_proba(items.values,users.values) 
		vpred = model.score(items, users)
		print(vpred)

		#### PCA
		#dimred_args = {"n_components": 2, "random_state": model.random_seed}
		#with np.errstate(invalid="ignore"): # for NaN or 0 variance matrices
		#	pca = PCA(**dimred_args)
		#	dimred_X = pca.fit_transform(model.model["feature_embeddings"])
		#	var12 = pca.explained_variance_ratio_[:2]*100
		#plt.xticks(fontsize=15)
		#plt.yticks(fontsize=15)
		#plt.xlabel("Var PC1 %.3f" % var12[0])
		#plt.ylabel("Var PC2 %.3f" % var12[1])
		#plt.grid(which='major', color='#D7D7D7', linewidth=2.0)
		#plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=1.0)
		#plt.minorticks_on()
		#plt.gca().spines[['top','right']].set_visible(False)
		#plt.scatter(dimred_X[:,0], dimred_X[:,1])
		#plt.savefig("features_figure.png",bbox_inches="tight")
		#plt.close()

		#gc.collect()
		
if __name__ == '__main__':
    unittest.main()
