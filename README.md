# ConsistencyAwareLib

ConsistencyAwareLib is a Python metrics library centered on the idea of **mitigating random consistency**. It is designed for classification evaluation, feature selection, correlation analysis, and representation learning. The current library includes the following metrics:

- `PureAccuracy`
- `StandardizedAccuracy`
- `SGINI`
- `PHSIC`
- `PSED`
- `MI`
- `NMI`
- `VI`
- `AMI`
- `MIq`
- `NMIq`
- `VIq`
- `NVIq`
- `AMIq`

## Installation

It is recommended to run the following command in the project root directory:

```bash
  pip install -e .
```

For a standard installation:

```bash
  pip install .
```

The command for installation via GitHub is as follows:

```bash
  pip install git+https://github.com/JietingWang91/ConsistencyAwareLib.git
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
| `MI(labels1, labels2)` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition | Returns a scalar. A larger value indicates more shared information between the two partitions; `0` means no mutual information under the Shannon definition. | Measures the amount of shared information between two partitions without normalization or correction for random consistency. |
| `NMI(labels1, labels2, method="sum")` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `method`: normalization choice in `{"min", "max", "sqrt", "sum", "joint"}` | Returns a scalar. A larger value indicates higher similarity between two partitions after normalization, which makes results easier to compare across settings. | Normalizes mutual information by an entropy-based upper bound to improve comparability across different clustering structures. |
| `VI(labels1, labels2)` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition | Returns a nonnegative scalar. A smaller value indicates that the two partitions are more similar; `0` indicates identical partitions. | Measures partition dissimilarity from an information-theoretic perspective and is useful when a distance-style metric is preferred. |
| `AMI(labels1, labels2, model="perm", method="sum", sided="two-sided")` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `model`: null model in `{"perm", "num", "all"}`; `method`: normalization choice in `{"min", "max", "sqrt", "sum"}`; `sided`: correction mode in `{"two-sided", "one-sided"}` | Returns a scalar. A larger value indicates stronger agreement after correcting for random consistency; a value close to `0` suggests similarity near the random baseline, and `1` corresponds to perfect agreement under the selected normalization. | Adjusts mutual information by subtracting the expected similarity under a chosen null model, reducing the inflation caused by random agreement. |
| `MIq(labels1, labels2, q)` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `q`: Tsallis entropic index (`q != 1`) | Returns a scalar. A larger value indicates more shared information under the generalized Tsallis entropy definition. | Extends mutual information to the Tsallis-entropy setting so that the sensitivity of the metric can be adjusted through `q`. |
| `NMIq(labels1, labels2, q, method="sum")` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `q`: Tsallis entropic index (`q != 1`); `method`: normalization choice in `{"min", "max", "sqrt", "sum", "joint"}` | Returns a scalar. A larger value indicates higher similarity after generalized normalization in the Tsallis setting. | Normalizes `MIq` by a Tsallis-entropy upper bound to make generalized mutual-information scores more interpretable and comparable. |
| `VIq(labels1, labels2, q)` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `q`: Tsallis entropic index (`q != 1`) | Returns a nonnegative scalar. A smaller value indicates that the two partitions are more similar under the generalized Tsallis definition; `0` indicates identical partitions. | Extends the variation-of-information distance to the Tsallis setting for generalized partition dissimilarity analysis. |
| `NVIq(labels1, labels2, q)` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `q`: Tsallis entropic index (`q != 1`) | Returns a nonnegative scalar. A smaller value indicates stronger similarity, while a value around `1` suggests that the dissimilarity is close to the random baseline under the permutation model. | Normalizes `VIq` by its expected value under random consistency, making the generalized distance easier to interpret against a random baseline. |
| `AMIq(labels1, labels2, q, method="sum")` | `labels1`: label sequence of the first partition; `labels2`: label sequence of the second partition; `q`: Tsallis entropic index (`q != 1`); `method`: normalization choice in `{"min", "max", "sqrt", "sum"}` | Returns a scalar. A larger value indicates stronger partition agreement after generalized random-consistency correction; a value close to `0` suggests similarity near the generalized random baseline. | Adjusts generalized mutual information in the Tsallis setting by removing the expected contribution caused by random consistency. |

## Usage Notes

- `PureAccuracy` and `StandardizedAccuracy` are suitable for evaluating classification prediction results.
- `SGINI` is suitable for decision tree node splitting and feature importance analysis.
- `PHSIC` is suitable for nonlinear correlation analysis, independence testing, and causal inference.
- `PSED` is suitable for representation learning, similarity learning, and objective design in deep model training.
- `MI` is suitable for measuring the raw amount of shared information between two partitions when normalization is not required.
- `NMI` is suitable for clustering comparison tasks where a normalized similarity score is preferred for comparison across datasets or experimental settings.
- `VI` is suitable for tasks that prefer a distance-style partition measure, where smaller values indicate more similar clustering results.
- `AMI` is suitable when clustering similarity should be evaluated after correcting for random agreement under a specified null model.
- `MIq` is suitable for generalized information analysis when the user wants to tune sensitivity through the Tsallis parameter `q`.
- `NMIq` is suitable for generalized clustering comparison when both Tsallis entropy and normalized similarity are desired.
- `VIq` is suitable for generalized distance-based comparison of partitions in the Tsallis setting.
- `NVIq` is suitable for interpreting generalized partition dissimilarity relative to its random baseline under the permutation model.
- `AMIq` is suitable for generalized clustering evaluation when the similarity score should be corrected for random consistency in the Tsallis setting.

## References (BibTeX)

The references for PureAccuracy、StandardizedAccuracy、SGINI、PHSIC、PSED are as follows:

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

If you have any questions, please contact jtwang@sxu.edu.cn.
