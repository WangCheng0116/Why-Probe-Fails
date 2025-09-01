<div align="center">
    <h1>Model–Task Alignment Drives Distinct RL Outcomes</h1>


[![arxiv](https://img.shields.io/badge/Arxiv-xxxx.xxxxx-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
</div>


## 📝 Overview  
Large Language Models (LLMs) can comply with harmful instructions, raising critical safety concerns.  
This project systematically re-examines probing-based methods for malicious input detection.  

Our study reveals that probing classifiers:  
- Achieve near-perfect accuracy in in-domain evaluations but collapse on out-of-distribution data.  
- Rely mainly on **instructional patterns** and **trigger words**, not true harmfulness semantics.  
- Create a *false sense of security*, highlighting the need for more principled safety detection approaches.  

---

## ⚙️ Reproduction  
We provide scripts to reproduce all key experiments:  

1. **Dataset preparation** – Download benign and malicious datasets.
3. **Hidden state extraction** – Collect internal representations from target LLMs.  
4. **Classifier training** – Train SVMs and Naive Bayes classifiers on the extracted representations.  
5. **Evaluation** – Benchmark across in-distribution, out-of-distribution, cleaned, and paraphrased datasets.  

---

## 📚 Citation  
If you find this work useful, please cite:  

```bibtex
@article{wang2025falsesense,

}
```
