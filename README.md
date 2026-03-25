# ConsistencyAwareLib

ConsistencyAwareLib is a Python metrics library centered on the idea of **mitigating random consistency**. It is designed for classification evaluation, feature selection, correlation analysis, and representation learning. The current library includes the following metrics:

- `PureAccuracy`
- `StandardizedAccuracy`
- `SGINI`
- `PHSIC`
- `PSED`

## Installation

It is recommended to run the following command in the project root directory:

```bash
  pip install -e .
```

For a standard installation:

```bash
  pip install .
```

## Dependencies

- Python >= 3.8
- numpy
- scipy
- torch = 2.8 (if using `PSED`)

## Metric Description

| Metric Name | Input Parameters | Meaning of the Output | Purpose of the Metric |
|---|---|---|---|
| `PureAccuracy(predicted_labels, true_labels, num_classes)` | `predicted_labels`: predicted label sequence; `true_labels`: ground-truth label sequence; `num_classes`: number of classes | Returns a scalar. A larger value indicates better classification performance after removing the effect of random consistency; a value close to 0 suggests that the result is more explainable by random consistency. | Removes the influence of random accuracy from ordinary accuracy, reducing the bias caused by class distribution and output preference. |
| `StandardizedAccuracy(predicted_labels, true_labels, num_classes)` | `predicted_labels`: predicted label sequence; `true_labels`: ground-truth label sequence; `num_classes`: number of classes | Returns a scalar. It reflects classification performance after standardized correction, making comparison across different tasks or conditions easier. | Further standardizes accuracy-based evaluation to improve comparability and robustness. |
| `SGINI(X, Y)` | `X`: attribute/feature vector or matrix; `Y`: label vector | Returns a scalar. A larger value usually indicates a stronger association between the feature and the label, and that this association is less likely to be caused only by randomness. | Mitigates the multivalue bias of the Gini index in decision trees and feature selection, reducing inflated evaluation caused by random consistency. |
| `PHSIC(X, Y, sX=None, sY=None, nrperm=0)` | `X`: sample matrix, typically of shape `(N, d1)`; `Y`: sample matrix, typically of shape `(N, d2)`; `sX`, `sY`: kernel bandwidths; `nrperm`: permutation parameter | Returns a scalar. A larger value usually indicates stronger dependence between variables; a value close to 0 indicates dependence close to the random baseline. | Subtracts the expected term under random conditions from HSIC, reducing spurious correlation in small-sample, high-dimensional, and noisy scenarios. |
| `PSED(predicted_features, true_labels)` | `predicted_features`: sample representation or feature matrix; `true_labels`: ground-truth label vector | Returns a scalar. When used as a loss, smaller is usually better, indicating that the learned sample similarity structure is closer to the true class structure. | Mitigates random consistency in similarity-matrix evaluation and can be used in representation learning and objective design for deep learning. |

## Usage Notes

- `PureAccuracy` and `StandardizedAccuracy` are suitable for evaluating classification prediction results.
- `SGINI` is suitable for decision tree node splitting and feature importance analysis.
- `PHSIC` is suitable for nonlinear correlation analysis, independence testing, and causal inference.
- `PSED` is suitable for representation learning, similarity learning, and objective design in deep model training.

## References (BibTeX)

```bibtex
@article{wang2020learning,
  title={Learning with mitigating random consistency from the accuracy measure},
  author={Wang, Jieting and Qian, Yuhua and Li, Feijiang},
  journal={Machine Learning},
  volume={109},
  number={12},
  pages={2247--2281},
  year={2020},
  publisher={Springer}
}

@ARTICLE{9765714,
  author={Wang, Jieting and Qian, Yuhua and Li, Feijiang and Liang, Jiye and Zhang, Qingfu},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Generalization Performance of Pure Accuracy and its Application in Selective Ensemble Learning}, 
  year={2023},
  volume={45},
  number={2},
  pages={1798-1816},
  keywords={Loss measurement;Particle measurements;Atmospheric measurements;Support vector machines;Phase change materials;Measurement uncertainty;Error probability;Generalization performance bound;linear-fractional measure;pure accuracy;selective ensemble learning},
  doi={10.1109/TPAMI.2022.3171436}
  }

@article{:/publisher/Science China Press/journal/SCIENTIA SINICA Informationis/54/1/10.1360/SSI-2022-0337,
  author = "Jieting WANG,Feijiang LI,Jue LI,Yuhua QIAN,Jiye LIANG",
  title = "Gini index and decision tree method with mitigating random consistency",
  journal = "SCIENTIA SINICA Informationis",
  year = "2024",
  volume = "54",
  number = "1",
  pages = "159-",
  url = "http://www.sciengine.com/publisher/Science China Press/journal/SCIENTIA SINICA Informationis/54/1/10.1360/SSI-2022-0337,
  doi = "https://doi.org/10.1360/SSI-2022-0337"
}

@inproceedings{ijcai2024p233,
  title     = {PHSIC against Random Consistency and Its Application in Causal Inference},
  author    = {Li, Jue and Qian, Yuhua and Wang, Jieting and Liu, Saixiong},
  booktitle = {Proceedings of the Thirty-Third International Joint Conference on
               Artificial Intelligence, {IJCAI-24}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Kate Larson},
  pages     = {2108--2116},
  year      = {2024},
  month     = {8},
  note      = {Main Track},
  doi       = {10.24963/ijcai.2024/233},
  url       = {https://doi.org/10.24963/ijcai.2024/233},
}


@inproceedings{
wang2025stabilizing,
title={Stabilizing Sample Similarity in Representation via Mitigating Random Consistency},
author={Jieting Wang and ZhangZelong and Feijiang Li and Yuhua Qian and Xinyan Liang},
booktitle={Forty-second International Conference on Machine Learning},
year={2025},
url={https://openreview.net/forum?id=FyiTRWgBR3}
}
```

## Contact

If you have any questions, please create an issue on this repository or contact at jtwang@sxu.edu.cn.
