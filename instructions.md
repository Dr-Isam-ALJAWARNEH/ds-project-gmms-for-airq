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
            > **N.B.** For both query sets, you need to perform spatial join between the point data (i.e., being it sensor air quality reading or taxi mobility in .csv format) and the polygons data (.geosjon file [available online](https://raw.githubusercontent.com/IsamAljawarneh/datasets/master/data/nyc_polygon.geojson)), then you need to perform `groupby` operation on either `neighborhood` or `geohash`, then you compare as per the example attached in `stratified_sampling_RMSE_mobility_NYC.ipynb` in the `code` folder.
            - Draw x-y graphs, where x is the new `sample size` and y is the `accuracy` for both types of the queries.
- [ ] You need then to capture the time-based QoS requirements (e.g., `running time` of each algorithm). You normally need to capture `end-to-end` `running time` for all stages of the processing pipeline, from `data cleaning` to `transformation`, to `generation` to `spatial join`, all the way down to `query processing`. Draw x-y graphs, where x is the new `sample size` and y is the `running time`
    
-------------------------------------------------
<!-- Task 3 -->
# [ ] Task 3! 
  # `TODO:` next 
- [ ] You need to test generated new `samples` data using a `density-based clustering` workload. Specifically the following:
    - you first need to complete the tutorial described in [dbscan](#dbscan-algorithm) in [task1](#--task-1), then you need to do the following task as described in sub-section [novel adapted DBSCAN](#dbscan-workload)

------------------------------------------------
# dbscan workload
- you need to `adapt` the stock version of the DBSCAN so that it operates a bit differently, specifically you need to do the following:
    - plain scikit-learn DBSCAN distance calculation relies on using  ```haversine``` as a distance metric to calculate haversine distances (i.e., geometrical distances) between coordinates (longitudes/latitudes pairs)
  > the attention that should be given in this case is that you did not capture any statistics regarding the distribution of the pm25 values (our target variable), you could for example capture the histograms of those values.
  Read more about possible values of ```pm2.5``` in this reference [PM2.5 particles in the air](https://www.epa.vic.gov.au/for-community/environmental-information/air-quality/pm25-particles-in-the-air). This means that you need to create a histogram showing the density of each bracket, your binning strategy should rely on the community definition of ranges of values. For example, binning example is the follwoing: Less than 25, 25–50, 50–100, 100–300, More than 300. 
    - Also, draw histograms showing the same binning and density of pm2.5 values in each neighborhood in your data. By the way, how many neighborhoods you have in your data?!
  > this is important as it will inform us about the fact whether nearby locations are having similar pm25 values. Why do we need to do this, because it is only in that case we consider those as a cluster, since they are geographiclly nearby, and also having simialr feature values (pm25 in this case). So what you need to do next is the following:
  - .. ```Extract and normalize several features```, similar to what has been done in the following tutorial, read specifically [Extending DBSCAN beyond just spatial coordinates](https://musa-550-fall-2020.github.io/slides/lecture-11A.html), thereafter ```Run DBSCAN to extract high-density clusters``` passing as an argument to the DBSCAN the new ```scaled features```, probably something like ```cores, labels = dbscan(features, eps=eps, min_samples=min_samples)``` . notice passing features (including pm25, longitude, latitude) instead of simply the coordinates.
  - having done this novel distance calculation (based on geometrical distance and pm25 values distance), calculate again the ```silhouette_score```, and check wether you obtain a higher accuracy (higher silhouette_score values) or not!

  > **N.B.** we are able to do this because of the definition of ```metric``` in DBSCAN which says **metric: The metric to use when calculating distance between instances in** a ```feature array``` so, it  a distance between several features is possible, given a ```feature array```, so put your scaled features in a feature array.
-------------------------------------------------
<!-- Task 3 -->
# [ ] Task 3! 
  # `TODO:` next
- apply everything you have done (and you need to do in `task2`) to a second dataset, probably `NYC taxi mobility data`, that is [available online](https://github.com/IsamAljawarneh/datasets/tree/master/data), `nyc1.zip`. the target variable is `trip_distance`. You need to capture `accuracy` and `running time` as described previously.

- numbers to capture
    - parameters : generated `data size`  in the x-axis , accuracy [RMSE](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/) and Mean Absoulte Percentage Error [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) for geo-statistics.  [RMSE equation](https://en.wikipedia.org/wiki/Root-mean-square_deviation). Same applies to `Top-N` queries.
--------------------------------------------

# [ ] Task 1
## `Partially Completed!` --> check the algorithms and the models!
- [X] Have a look at the Example: GMM for Generating New Data [Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html). AND
- [X] Kernel Density Estimation [KDE](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)
- [X] the data file contains granular (meter-meter level) low-cost air quality data, measuring most importantly Particulate Matters (PM10 and PM2.5), in addition to some other features including location represented as longitudes/latitudes!
- [X] compare the performance of GMM with several other generative algorithms such as few of the following:
    - Kernel Density Estimation (KDE): KDE is a non-parametric method that estimates the probability density function of a random variable based on its observed data points. It works by placing a kernel function on each data point and summing them up to obtain the estimated density. To generate new data, one can sample from the estimated density function.
    - Generative Adversarial Networks (GANs): GANs are deep learning models that consist of two neural networks, a generator and a discriminator, which are trained simultaneously. The generator learns to generate data samples that are indistinguishable from real data, while the discriminator learns to distinguish between real and generated data. GANs can be trained on historical data to generate new data points.
    - Variational Autoencoders (VAEs): VAEs are another type of generative model that learn a low-dimensional representation of the input data and generate new data points by sampling from this learned representation. VAEs consist of an encoder network that maps input data to a latent space and a decoder network that reconstructs the input data from the latent space. By sampling from the latent space, new data points can be generated.
    - Mixture Density Networks (MDNs): MDNs are neural networks that model the conditional probability distribution of the output variable given the input data as a mixture of Gaussian distributions. Similar to GMMs, MDNs can capture complex multi-modal distributions in the data and generate new data points by sampling from the learned mixture distribution.
    - Copula-based Models: Copulas are statistical tools that describe the dependence structure between random variables separately from their marginal distributions. Copula-based models can be used to generate new data by sampling from the copula function while specifying the marginal distributions of the variables.
# dbscan algorithm

- [ ] `you need to complete this part' - DBSCAN

        1. [ ] run the example clustering code (DBSCAN), attached in the new folder "DBSCAN_Clustering" inside 'starting_code' folder
        2. [ ] read more about how DBSCAN works in scikit-learn
            > [DBSCAN scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)
        3. [ ] you need to use ```Silhouette Coefficient ```for evaluation
            > ***"If the ground truth labels are not known, evaluation can only be performed using the model results itself. In that case, the Silhouette Coefficient comes in handy."***
            an example is here:
            [Demo of DBSCAN clustering algorithm](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py)

