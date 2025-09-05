<div align="center">
    <h1>False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize</h1>


[![arxiv](https://img.shields.io/badge/Arxiv-2509.03888-b31b1b.svg?logo=arXiv)](https://arxiv.org/pdf/2509.03888) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT) 
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

1. **Dataset preparation** – We have provided the dataset in `data` folder.
2. **Hidden state extraction** – Collect internal representations from target LLMs using `get_hidden_states.py`
3. **Classification** – Train SVMs classifiers on the extracted representations and evaluate the results using `classify.py`
 

---

## 📚 Citation  
If you find this work useful, please cite:  

```bibtex
@misc{wang2025falsesensesecurityprobingbased,
      title={False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize}, 
      author={Cheng Wang and Zeming Wei and Qin Liu and Muhao Chen},
      year={2025},
      eprint={2509.03888},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.03888}, 
}
```
