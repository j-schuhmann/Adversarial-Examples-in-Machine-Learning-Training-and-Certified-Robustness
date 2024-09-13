# Adversarial-Examples-and-Robustness-in-Machine-Learning

This repository contains code for my master thesis: "Adversarial Examples and Robustness in Machine Learning"

### Overview

- **adversarial_training**: Includes normal training and further PGD and FGSM adversarial training with the $\ell^\infty$-norm and the $\ell^2$-norm.
- **attacks**: Includes PGD and FGSM adversarial attacks with the $\ell^\infty$-norm and the $\ell^2$-norm.
- **classified_robustness**: Includes the code for the classified robustness part.
- **models**: Contains the different models used in the experiments.
- **util**: Contains utility functions and helper code used across the project. 



### Dependencies

The following packages were used in this project:

- `numpy==1.25.2`
- `tensorflow==2.16.1`
- `scipy==1.11.4`