# DRIFT: Drift-Resilient Invariant-Feature Transformer for DGA Detection

**`This model corresponds to the final development version at the time of manuscript submission.`**

**DRIFT** is a specialized deep learning framework designed to solve the problem of **concept drift** in Domain Generation Algorithm (DGA) detection. While traditional detectors perform well in static environments, their accuracy often degrades over time as attackers evolve their generation logic. This project introduces a **dual-branch Transformer** architecture that learns invariant structural features to ensure long-term dependability in evolving threat landscapes.

---

## 1. Key Components

* **Hybrid Tokenization Strategy**: The model simultaneously processes domain names through two different lenses to capture heterogeneous generation patterns:
* **Character-level Encoding**: Captures stochastic morphological patterns found in random-string DGAs.
* **Subword-level Encoding**: Uses the WordPiece algorithm to model semantic regularities in dictionary-based DGAs.


* **Multi-Task Self-Supervised Pre-training**: To improve robustness without requiring massive labeled datasets, the model is pre-trained on three auxiliary tasks:
1. **Masked Token Prediction (MTP)**: Learns bidirectional context by predicting hidden tokens.
2. **Token Position Prediction (TPP)**: Learns global structure by reconstructing the original order of shuffled tokens.
3. **Token Order Verification (TOV)**: Learns high-level coherence by discriminating between original and scrambled sequences.


* **Domain-Name-Only Detection**: The system functions strictly using the domain string itself, remaining effective even when external signals like DNS response anomalies or OSINT are unavailable.

---

## 2. Model Architecture

The framework utilizes a dual-branch Transformer encoder where each branch is pre-trained independently and then fused for final classification.

### Loss Function

The total pre-training loss is calculated by summing the losses from the three subtasks:


### Feature Fusion

Information is aggregated from the final hidden states using a combination of **Max Pooling** and **Mean Pooling** across both the subword and character branches.

---

## 3. Experimental Performance

A longitudinal study spanning nine years (2017–2025) was conducted to evaluate performance under "forward-chaining," where models are tested on data from future years not seen during training.

> Comprehensive evaluations demonstrate that our method significantly mitigates temporal degradation and consistently outperforms state-of-the-art baselines in forward-chaining experiments. The proposed approach offers a dependable foundation for long-term DGA defense in evolving threat landscapes.

In our comparative analysis, we reproduced the MIT and NYU models from Yu et al. (2018) to serve as primary benchmarks for character-level and hybrid DGA detection. These models were evaluated against a diverse set of state-of-the-art methodologies—including Endgame (2016), B-ResNet (2020), M-ResNet + B-cos (2022), Dom2Vec (2023), HMT (2023), SFT-Llama3-8B (2024), and HDDN (2025)—under a forward-chaining protocol to quantify their resilience to concept drift over a nine-year longitudinal study. Since the submission of the paper, we are currently evolving the proposed method to further enhance its performance.

---

## 4. Future Roadmap

1. **Architecture Optimization**: Refining the pretrain logic and dual-branch fusion mechanism.
2. **Field Deployment**: Testing the model in real-time network traffic environments.



---

## Authors

* **Chaeyoung Lee** (BS Artificial Intelligence Engineering '27 @ Sookmyung Women's Univ. | SNSec. Lab)


* **Chaeri Jung** (BS Artificial Intelligence Engineering '27 @ Sookmyung Women's Univ. | SNSec. Lab)

