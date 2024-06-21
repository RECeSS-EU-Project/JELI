![funding logo](https://raw.githubusercontent.com/RECeSS-EU-Project/RECeSS-EU-Project.github.io/main/assets/images/header%2BEU_rescale.jpg)

# Joint Embedding-classifier Learning for improved Interpretability (JELI) Python Package

This repository is a part of the EU-funded [RECeSS project](https://recess-eu-project.github.io) (#101102016), and hosts the code for the open-source Python package *JELI* for the collaborative filtering approach.

[![Python Version](https://img.shields.io/badge/python-3.8%7C3.9-pink)](https://img.shields.io/badge/python-3.8%7C3.9-pink) ![GitHub](https://img.shields.io/github/license/recess-eu-project/jeli.svg) [![Build Status](https://github.com/recess-eu-project/jeli/actions/workflows/post-push-test.yml/badge.svg)](https://github.com/recess-eu-project/jeli/actions/workflows/post-push-test.yml) [![Codecov](https://codecov.io/github/recess-eu-project/jeli/coverage.svg?branch=master)](https://codecov.io/github/recess-eu-project/jeli?branch=master) [![Codefactor](https://www.codefactor.io/repository/github/recess-eu-project/jeli/badge?style=plastic)](https://www.codefactor.io/repository/github/recess-eu-project/jeli) 

## Statement of need 

Interpretability is a topical question in recommender systems, especially in healthcare applications. In drug repurposing, the goal is to identify novel therapeutic indications as drug-disease pairs. An interpretable drug repurposing algorithm quantifies the importance of each input feature for the predicted therapeutic drug-disease association in a non-ambiguous fashion, using post hoc methods. Unfortunately, different importance score-based approaches lead to different results, yielding unreliable interpretations.

We introduce the novel Joint Embedding Learning-classifier for improved Interpretability (JELI). It features a new structured recommender system and trains it jointly on a drug-disease-gene knowledge graph completion task. In particular, JELI simultaneously (a) learns the gene, drug, and disease embeddings; (b) predicts new drug-disease associations based on those embeddings; (c) provides importance scores for each gene. The drug and disease embeddings have a structure that depends on the gene embeddings. Therefore, JELI allows the introduction of graph-based priors on the connections between diseases, drugs, and genes in a generic fashion to recommend and argue for novel therapeutic drug-disease associations. 

Contrary to prior works, the recommender system explicitly includes the importance scores, strengthening the link between the recommendations and the extracted scores while allowing the use of a generic embedding model. The recommendation strategy in JELI can also be readily applied beyond the task of drug repurposing for any sets of items, users, and features.

## Install the latest release

### Using pip

```bash
pip install jeli
```

### Docker

```bash
#Build Docker image
docker build -t jeli .
#Run Docker image built in previous step and drop into SSH
docker run -it --expose 3000  -p 3000:3000 jeli
```

### Dependencies

OS: developed and tested on Debian Linux.

The complete list of dependencies for *JELI* can be found at [requirements.txt](https://raw.githubusercontent.com/RECeSS-EU-Project/JELI/master/pip/requirements.txt) (pip).

## Licence

This repository is under an [OSI-approved](https://opensource.org/licenses/) [MIT license](https://raw.githubusercontent.com/RECeSS-EU-Project/JELI/master/LICENSE). 

## Citation

If you use *JELI* in academic research, please cite it as follows

```
Réda, Clémence, Vie, Jill-Jênn and Wolkenhauer, Olaf (2024). JELI: an interpretable embedding-learning recommender system for drug repurposing.
```

## Community guidelines with respect to contributions, issue reporting, and support

[Pull requests](https://github.com/RECeSS-EU-Project/JELI/pulls) and [issue flagging](https://github.com/RECeSS-EU-Project/JELI/issues) are welcome, and can be made through the GitHub interface. Support can be provided by reaching out to ``recess-project[at]proton.me``. However, please note that contributors and users must abide by the [Code of Conduct](https://github.com/RECeSS-EU-Project/JELI/blob/master/CODE%20OF%20CONDUCT.md).

