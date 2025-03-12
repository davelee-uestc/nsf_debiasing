# Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness of Samples

This repository contains experiments for the paper _Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness of Samples_ (CVPR 2025)

[[Project]](https://davelee-uestc.github.io/nsf_cvpr25) [[Paper]](https://cvpr.thecvf.com/virtual/2025/poster/34895)

## Bibtex

```
@inproceedings{li2024let,
            title={Let Samples Speak: Mitigating Spurious Correlation by Exploiting the Clusterness of Samples},
            author={Weiwei Li and Junzhuo Liu and Yuanyuan Ren and Yuchen Zheng and Yahao Liu and Wen Li},
            booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
            year={2025},
          }
```

## Introduction
Deep learning models are known to often learn spurious features that correlate with the label during training but are irrelevant to the prediction task. Existing methods typically address this issue by annotating potential spurious attributes, or filtering spurious features based on some empirical assumptions (e.g., simplicity of bias). However, these methods may yield unsatisfying performance due to the intricate and elusive nature of spurious correlations in real-world data. In this paper, we propose a data-oriented approach to mitigate the spurious correlation in deep learning models. We observe that samples that are influenced by spurious features tend to exhibit a dispersed distribution in the learned feature space. This allows us to identify the presence of spurious features. Subsequently, we obtain a bias-invariant representation by neutralizing the spurious features based on a simple grouping strategy. Then, we learn a feature transformation to eliminate the spurious features by aligning with this bias-invariant representation. Finally, we update the classifier by incorporating the learned feature transformation and obtain an unbiased model. By integrating the aforementioned identifying, neutralizing, eliminating and updating procedures, we build an effective pipeline for mitigating spurious correlation. Experiments on four image and NLP debiasing benchmarks and one medical dataset demonstrate the effectiveness of our proposed approach, showing an improvement of worst-group accuracy by over 20% compared to standard empirical risk minimization (ERM). 
## File Structure

```
.
+-- train_supervised.py (Train base models)
+-- ssc.py (Main Experiments)
+-- ssc_common.py (Common utilities used in different scripts)
```

## Requirements

- [`torch`](https://pytorch.org/get-started/locally/)
- [`torchvision`](https://pytorch.org/get-started/locally/)
- [`timm`](https://github.com/rwightman/pytorch-image-models)
- [`transformers`](https://huggingface.co/docs/transformers/installation)
- [`vissl`](https://github.com/facebookresearch/vissl/blob/main/INSTALL.md)
- [`scikit-learn`](https://scikit-learn.org/stable/install.html)
- [`numpy`](https://numpy.org/install/)
- [`tqdm`](https://pypi.org/project/tqdm/)
- [`wilds`](https://wilds.stanford.edu/get_started/)


## Data access

### Waterbirds and CelebA

Please follow the instructions in the [DFR repo](https://github.com/PolinaKirichenko/deep_feature_reweighting#data-access) to prepare the Waterbirds and CelebA datasets in the `./waterbirds`.

### Civil Comments and MultiNLI

The Civil Comments dataset should be downloaded automatically when you run experiments, no manual preparation needed.

To run experiments on the MultiNLI dataset, please manually download and unzip the dataset from [this link](https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz) in the `./multinli`.
Further, please copy the `dataset_files/utils_glue.py` to the root directory of the dataset.

### CheXpert

The CheXpert dataset are not publically available, so we share the code `medical_dataset_preprocess.py` for preparing this dataset.


## Example commands

Waterbirds:
```bash
./train_sup.sh -d wb
python3 ssc.py waterbirds 
```

CelebA:
```bash
./train_sup.sh -d ce
python3 ssc.py celeba 
```


Civil Comments
```bash
./train_sup.sh -d cc
python3 ssc.py multinli 
```

MultiNLI
```bash
./train_sup.sh -d mn
python3 ssc.py civilcomments 

```

CheXpert
```bash
./train_sup.sh -d cx
python3 ssc.py chexpert 

```


## Code References

- We used the [DFR codebase](https://github.com/izmailovpavel/spurious_feature_learning) as the basis for our code.

- Our model implementations are based on the `torchvision`, `timm` and `transformers` packages.
