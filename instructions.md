# instructions
## some insights:
---------------------------------------
<!-- Task 2 -->
# [ ] Task 2! 
# `Update: April 7, 2024`
### `N.B.` references are available in the end of this instruction file!
# `Required OPTIMIZATION ==> IMPORTANT!`
## `TODO:` 
- [ ] Using the algorithms that you have developed, (being it GMM, KDE, MDN), 
generate sufficient amount of new `data samples`. The size of the `generated` data should be the same as the size of the `original` data that has been used for training for generating the data. You should generate sufficient amount of `new samples` to be able to perform subsequent tests. For example, generate 100K (one hundred thousands) new samples. Thereafter, you need to test the accuracy and the quality of the generated data based on several query workloads as follows (**N.B.** first, for the NYC Air Quality (AQ) data, the target variable is `pm25`)
    - geo-statistics queries (e.g., `count` and `average`). Example query is `"what is the average pm25 each neighborhood of NYC"`, OR `"what is the pm25 readings count in each neighborhood in NYC"`. 
        - those queries can be tested using the RMSE, and MAPE.
    - Top-N. Example query: "what are the Top-N neighborhoods in NYC in terms of pm25 values"
        - those queries can be tested using Pearson, [jensen shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) [Kullback–Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) and other metrics. Attached example notebook on `NYC taxi mobility data`
            > to calculate any of these metrics to test the accuracy, you need to group data into two distributions, one for the `original` and one for the `generated` data, then calculate those metrics as distances between distributions.
            > I have attached an example coding for calculating `RMSE`, `MAPE`, `Pearson`, `KL Divergence` and other metrics for the count and Top-N, respectively. That is to say, how much is the difference in count of `taxi` trips between the original data and the sampled data for each neighborhood in New York City (NYC). Attached example notebook titled `stratified_sampling_RMSE_mobility_NYC.ipynb` in the `code` folder.
--------------------------------------
<!-- Task 1 -->
# [ ] Task 1! 
- [X] Have a look at the Example: GMM for Generating New Data [Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html). AND
- [X] Kernel Density Estimation [KDE](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)
- [X] the data file contains granular (meter-meter level) low-cost air quality data, measuring most importantly Particulate Matters (PM10 and PM2.5), in addition to some other features including location represented as longitudes/latitudes!
- [X] compare the performance of GMM with several other generative algorithms such as few of the following:
    - Kernel Density Estimation (KDE): KDE is a non-parametric method that estimates the probability density function of a random variable based on its observed data points. It works by placing a kernel function on each data point and summing them up to obtain the estimated density. To generate new data, one can sample from the estimated density function.
    - Generative Adversarial Networks (GANs): GANs are deep learning models that consist of two neural networks, a generator and a discriminator, which are trained simultaneously. The generator learns to generate data samples that are indistinguishable from real data, while the discriminator learns to distinguish between real and generated data. GANs can be trained on historical data to generate new data points.
    - Variational Autoencoders (VAEs): VAEs are another type of generative model that learn a low-dimensional representation of the input data and generate new data points by sampling from this learned representation. VAEs consist of an encoder network that maps input data to a latent space and a decoder network that reconstructs the input data from the latent space. By sampling from the latent space, new data points can be generated.
    - Mixture Density Networks (MDNs): MDNs are neural networks that model the conditional probability distribution of the output variable given the input data as a mixture of Gaussian distributions. Similar to GMMs, MDNs can capture complex multi-modal distributions in the data and generate new data points by sampling from the learned mixture distribution.
    - Copula-based Models: Copulas are statistical tools that describe the dependence structure between random variables separately from their marginal distributions. Copula-based models can be used to generate new data by sampling from the copula function while specifying the marginal distributions of the variables.

