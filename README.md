# 🧠 GCN + LLM for Enhanced Virtual Screening

This repository contains the code and experiments for combining Graph Convolutional Networks (GCNs) with Large Language Model (LLM) chemical knowledge to improve virtual screening. This work is based on the paper:

**"Combining GCN Structural Learning with LLM Chemical Knowledge for Enhanced Virtual Screening"**
📝 *Submitted to the Journal of Chemical Information and Modeling (JCIM)*

---

## 📘 Abstract

Virtual screening plays a critical role in modern drug discovery by enabling the identification of promising candidate molecules for experimental validation. Traditional machine learning methods such as support vector machines (SVM) and XGBoost rely on predefined molecular representations, often leading to information loss and potential bias. In contrast, deep learning approaches—particularly Graph Convolutional Networks (GCNs)—offer a more expressive and unbiased alternative by operating directly on molecular graphs. Meanwhile, Large Language Models (LLMs) have recently demonstrated state-of-the-art performance in drug design, thanks to their capacity to capture complex chemical patterns from large-scale data via attention mechanisms.

In this paper, we propose a hybrid architecture that integrates GCNs with LLM-derived embeddings to combine localized structural learning with global chemical knowledge. The LLM embeddings can be precomputed and stored in a molecular feature library, removing the need to rerun the LLM during training or inference and thus maintaining computational efficiency. We found that concatenating the LLM embeddings after each GCN layer—rather than only at the final layer—significantly improves performance, enabling deeper integration of global context throughout the network. The resulting model achieves superior results on the **[Name of Dataset(s)] dataset(s)**, with an F1-score of **88.8%**, outperforming standalone GCN (**87.9%**), XGBoost (**85.5%**), and SVM (**85.4%**) baselines.

---

## 🔍 Motivation

Drug discovery is often hindered by the trade-off between computational efficiency and chemical accuracy. This work aims to bridge this gap by combining the strengths of graph-based and language-based representations of molecules.

---

## 🚀 Key Contributions

- ✅ A novel hybrid GCN + LLM architecture for virtual screening.
- ⚡ Precomputed LLM embeddings to enhance efficiency.
- 🔗 Fusion strategy: concatenating LLM embeddings at each GCN layer.
- 📊 Benchmarked on **[Name of Dataset(s)]** with superior performance over traditional ML baselines.

---

## 🏗️ Project Structure

```.
├── main.py             # Entry point for training/testing
├── data/
│   ├── raw/            # Raw .sdf or .csv datasets
│   └── processed/      # Processed graphs & embeddings
├── src/
│   ├── preprocessing/  # Feature extraction, graph building
│   ├── models/         # GCN, hybrid model architecture
│   ├── encoding/       # LLM embedding module
│   └── utils/          # Helper functions
├── results/            # Evaluation reports, metrics
├── notebooks/          # Optional Jupyter/Colab analysis
├── requirements.txt
└── README.md
```


---

## 🧪 Reproducibility

```bash
# Step 1: Install dependencies
pip install -r requirements.txt

# Step 2: Download and prepare data
python src/preprocessing/prepare_data.py

# Step 3: Precompute LLM embeddings
python src/encoding/compute_embeddings.py

# Step 4: Train the model
python main.py --config configs/hybrid_model.yaml

# Step 5: Evaluate the model
python main.py --mode test --checkpoint path/to/model.ckpt

```
---

## 📊 Results
```
bash
Model | F1-Score
SVM | 85.4%
XGBoost | 85.5%
GCN | 87.9%
GCN + LLM | 88.8%

```

---


##📚 Citation

@article{your2025gcnllm,
  title={Combining GCN Structural Learning with LLM Chemical Knowledge for Enhanced Virtual Screening},
  author={Your Name and Collaborators},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  note={Under Review}
}



---


##📦 Dependencies
Python 3.10

RDKit 2023.09.1

PyTorch 2.2.0

PyTorch Geometric 2.4.0

Transformers 4.38

scikit-learn 1.4+



---


##📬 Contact
For questions, feedback, or collaboration proposals:

📧 rberreziga@usthb.dz
🔗 LinkedIn
📂 Institution/Lab (if applicable)