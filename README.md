# Debiasing_Variational_Autoencoder
In the second repository, we'll explore two prominent aspects of applied deep learning: facial detection and algorithmic bias.

Deploying fair, unbiased AI systems is critical to their long-term acceptance. Consider the task of facial detection: given an image, is it an image of a face? This seemingly simple, but extremely important, task is subject to significant amounts of algorithmic bias among select demographics.

In this model development, we'll investigate [one recently published approach to addressing algorithmic bias](http://introtodeeplearning.com/AAAI_MitigatingAlgorithmicBias.pdf) . We'll build a facial detection model that learns the latent variables underlying face image datasets and uses this to adaptively re-sample the training data, thus mitigating any biases that may be present in order to train a debiased model.


