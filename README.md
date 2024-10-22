# GenePert
GenePert: Leveraging GenePT Embeddings for Gene Perturbation Prediction


Gene perturbation screens, such as CRISPR-based perturbation screens, have enabled researchers to probe regulatory networks and decipher genetic causal effects at an increasing scale. This capability has sparked much interest in developing machine learning models to predict experimental outcomes that could not be easily gathered experimentally. 

We present a simple approach, `GenePert`, that leverages GenePT embeddings gene embeddings derived from biomedical literature descriptions of individual genes to predict gene expression outcomes following perturbations via penalized regression models. Benchmarked on nine perturbation screen datasets across multiple cell lines and five different pretrained gene embedding models, our approach performs competitively compared to state-of-the-art prediction models. Even with limited training data, our model generalizes effectively, offering a scalable solution for predicting perturbation outcomes. These findings underscore the power of gene embeddings in predicting the outcomes of unseen genetic perturbation experiments in silico.







