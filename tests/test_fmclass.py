import unittest
import numpy as np
import random
from stanscofi.datasets import generate_dummy_dataset, Dataset
from stanscofi.utils import load_dataset
from stanscofi.training_testing import random_simple_split
from stanscofi.validation import AUC, NDCGk, compute_metrics

import sys
sys.path.insert(0,"../src/")

from jeli.JELIImplementation import RHOFM
from jeli.JELI import JELI

class TestFMClass(unittest.TestCase):

	def test_rhofm(self):

		frozen = False
		cuda_on = False
		SEED = 1245
		TEST_SIZE = 0.2
		DATA_NAME = "Synthetic"
		DFOLDER="../datasets/"
		
		if (DATA_NAME=="Synthetic"):
			npositive, nnegative, nfeatures, mean, std = 200, 200, 100, 2, 0.01
			data_args = generate_dummy_dataset(npositive, nnegative, nfeatures, mean, std, random_state=SEED)
		else:
			data_args = load_dataset(DATA_NAME, save_folder=DFOLDER)
			data_args["name"] = DATA_NAME
		
		N_DIMENSIONS = 20
		N_EPOCHS=25
		BATCH_SIZE=1024
		
		np.random.seed(SEED)
		random.seed(SEED)

		## Import dataset
		dataset = Dataset(**data_args)
		
		(train_folds, test_folds), _ = random_simple_split(dataset, TEST_SIZE, metric="cosine", random_state=SEED)
		train = dataset.subset(train_folds)
		test = dataset.subset(test_folds)

		## Learn without the help of a KG
		print("\n----------- RHOFM")
		RHOFMmodel = RHOFM(dataset.nitem_features, 2, "linear", frozen, cuda_on)
		RHOFMmodel.fit(train, embeddings=None, n_epochs=N_EPOCHS, loss="CrossEntropyLoss", batch_size=BATCH_SIZE, random_seed=SEED)
		scores = RHOFMmodel.predict_proba(test)
		predictions = RHOFMmodel.predict(scores, threshold=0.5)
		JELI().print_scores(scores)

		metrics, _ = compute_metrics(scores, predictions, test, metrics=["AUC", "NDCGk"], k=dataset.nitems, beta=1, verbose=False)
		print(metrics)
			
if __name__ == '__main__':
    unittest.main()
