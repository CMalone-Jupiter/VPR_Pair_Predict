# Complementary Visual Place Recognition Technique Selection

Currently the code uploaded is research quality code. We are working on making it more accessible and readable when possible.

### 'Boosting Performance of a Baseline Visual Place Recognition Technique by Predicting the Maximally Complementary Technique' \[[IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10161561)]  \[[arXiv](https://arxiv.org/abs/2210.07509)]

### Abstract
One recent promising approach to the Visual Place Recognition (VPR) problem has been to fuse the place recognition estimates of multiple complementary VPR techniques using methods such as shared representative appearance learning (SRAL) and multi-process fusion. These approaches come with a substantial practical limitation: they require all potential VPR methods to be brute-force run before they are selectively fused. The obvious solution to this limitation is to predict the viable subset of methods ahead of time, but this is challenging because it requires a predictive signal within the imagery itself that is indicative of high performance methods. Here we propose an alternative approach that instead starts with a known single base VPR technique, and learns to predict the most complementary additional VPR technique to fuse with it, that results in the largest improvement in performance. The key innovation here is to use a dimensionally reduced difference vector between the query image and the top-retrieved reference image using this baseline technique as the predictive signal of the most complementary additional technique, both during training and inference. We demonstrate that our approach can train a single network to select performant, complementary technique pairs across datasets which span multiple modes of transportation (train, car, walking) as well as to generalise to unseen datasets, outperforming multiple baseline strategies for manually selecting the best technique pairs based on the same training data.

![Error](https://github.com/CMalone-Jupiter/VPR_Pair_Predict/blob/main/imgs/front_page_fig.png)

### Cite
If this repository contributes to your research, please consider citing the publication below.
```
C. Malone, S. Hausler, T. Fischer and M. Milford, "Boosting performance of a baseline visual place recognition technique by predicting the maximally complementary technique," in the IEEE International Conference on Robotics and Automation, May 2023.
```
#### Bibtex
```
@inproceedings{malone2023boosting,
  title={Boosting performance of a baseline visual place recognition technique by predicting the maximally complementary technique},
  author={Malone, Connor and Hausler, Stephen and Fischer, Tobias and Milford, Michael},
  booktitle={2023 IEEE International Conference on Robotics and Automation (ICRA)},
  pages={1919--1925},
  year={2023},
  organization={IEEE}
}
```