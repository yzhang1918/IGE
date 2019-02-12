# IGE


This is a PyTorch implementation of the __Interaction Graph Embedding (IGE)__ model proposed in [Zhang et al. (2017)](https://dl.acm.org/citation.cfm?id=3132918).

We also improved the model and developed the __IGE+__. The code will be available when the paper is accepted.

# Requirements

- python3
- pytorch
- pandas
- numpy
- scikit-learn

# Usage

```
python -m explib.experiments.exp_ige_dblp -c explib/configs/exp_ige_dblp.json
```


Note: Due to the large file size, we use a subset of DBLP in this example where only labeled authors and their records are kept.

__For your own datasets, please implement the dataloader and experiment files.__


# Cite

Please cite our paper if you make advantage of the IGE in your own work:

```bibtex
@inproceedings{zhang2017learning,
  title={Learning Node Embeddings in Interaction Graphs},
  author={Zhang, Yao and Xiong, Yun and Kong, Xiangnan and Zhu, Yangyong},
  booktitle={Proceedings of the 2017 ACM on Conference on Information and Knowledge Management},
  pages={397--406},
  year={2017},
  organization={ACM}
}
```



