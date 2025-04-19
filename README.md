# GCN + LLM for Enhanced Virtual Screening

This repository contains the code and experiments from the paper:  
**"Combining GCN Structural Learning with LLM Chemical Knowledge for Enhanced Virtual Screening"**

## 📘 Abstract

Virtual screening plays a critical role in modern drug discovery by enabling the identification of promising candidate molecules for experimental validation. Traditional machine learning methods such as support vector machines (SVM) and XGBoost rely on predefined molecular representations, often leading to information loss and potential bias. In contrast, deep learning approaches—particularly Graph Convolutional Networks (GCNs)—offer a more expressive and unbiased alternative by operating directly on molecular graphs. Meanwhile, Large Language Models (LLMs) have recently demonstrated state-of-the-art performance in drug design, thanks to their capacity to capture complex chemical patterns from large-scale data via attention mechanisms.

In this paper, we propose a hybrid architecture that integrates GCNs with LLM-derived embeddings to combine localized structural learning with global chemical knowledge. The LLM embeddings can be precomputed and stored in a molecular feature library, removing the need to rerun the LLM during training or inference and thus maintaining computational efficiency. We found that concatenating the LLM embeddings after each GCN layer—rather than only at the final layer—significantly improves performance, enabling deeper integration of global context throughout the network. The resulting model achieves superior results, with an F1-score of (88.8\%) , outperforming standalone GCN (87.9\%), XGBoost (85.5\%), and SVM (85.4\%) baselines. 


## 📁 Project Structure

