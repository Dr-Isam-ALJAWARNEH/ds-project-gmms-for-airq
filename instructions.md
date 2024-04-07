# instructions
---------------------------------------
<!-- Task 2 -->
# [ ] Task 2
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
# [ ] Task 3
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
<!-- Task 4 -->
# [ ] Task 4
  # `TODO:` next
- apply everything you have done (and you need to do in `task2`) to a second dataset, probably `NYC taxi mobility data`, that is [available online](https://github.com/IsamAljawarneh/datasets/tree/master/data), `nyc1.zip`. the target variable is `trip_distance`. You need to capture `accuracy` and `running time` as described previously.

- numbers to capture
    - parameters : generated `data size`  in the x-axis , accuracy [RMSE](https://www.statisticshowto.com/probability-and-statistics/regression-analysis/rmse-root-mean-square-error/) and Mean Absoulte Percentage Error [MAPE](https://en.wikipedia.org/wiki/Mean_absolute_percentage_error) for geo-statistics.  [RMSE equation](https://en.wikipedia.org/wiki/Root-mean-square_deviation). Same applies to `Top-N` queries.
--------------------------------------------
<!-- Task 5 -->
# [ ] Task 5

# `IMPortant` test with more than one data, add NYC taxi mobility data (for journal paper, you need tests on more than one data):
[available online](https://github.com/IsamAljawarneh/datasets/tree/master/data), `nyc1.zip`
- start writing your paper, either for conferences or journal. For journal, use the `applied sciences` template atatched in the `target-venue` folder titled `applsci-template.dot` or ```IEEE``` template attached, or other journal template that i can attach later on. (minimum 10 pages)
- or even, consider one of the following two conference (IEEE template for those conferences is attached) (minimum 6 pages)
    - [MCNA - Spain](https://mcna-conference.org/2024/committee.php)
    - [IDSTA - Croatia](https://idsta-conference.org/2024/calls.php)
1.  [ ] Include (cite appropriately) all of the following papers! reference papers include:
    # Category A : for sampling desing and Approximate Query Processing (AQP)
    > Spatial-aware approximate big data stream processing [^2] and
    > Polygon Simplification for the Efficient Approximate Analytics of Georeferenced Big Data [^3]
    > QoS-aware approximate query processing for smart cities spatial data streams. [^4]
    > Spatially representative online Big Data sampling for smart cities. [^5]
    # Category B: for spatial join procesing
    > SpatialSSJP: QoS-Aware Adaptive Approximate Stream-Static Spatial Join Processor [^6]
    > Efficient Integration of Heterogeneous Mobility-Pollution Big Data for Joint Analytics at Scale with QoS Guarantees [^7]
    > Efficiently integrating mobility and environment data for climate change analytics.[^8]
    > Efficient QoS-aware spatial join processing for scalable NoSQL storage frameworks. [^9]
    # Category C: clustering DBSCAN
    > Efficient spark-based framework for big geospatial data query processing and analysis [^10]
 
    [^1]: Al Jawarneh, I. M., Foschini, L., & Corradi, A. (2023, November). Efficient Generation of Approximate Region-based Geo-maps from Big Geotagged Data. In 2023 IEEE 28th International Workshop on Computer Aided Modeling and Design of Communication Links and Networks (CAMAD) (pp. 93-98). IEEE.
    [^2]: Al Jawarneh, I. M., Bellavista, P., Foschini, L., & Montanari, R. (2019, December). Spatial-aware approximate big data stream processing. In 2019 IEEE global communications conference (GLOBECOM) (pp. 1-6). IEEE. [available online](https://www.researchgate.net/profile/Isam-Al-Jawarneh/publication/339562314_Spatial-Aware_Approximate_Big_Data_Stream_Processing/links/5ff45764299bf14088708888/Spatial-Aware-Approximate-Big-Data-Stream-Processing.pdf)
    [^3]: Al Jawarneh, I. M., Foschini, L., & Bellavista, P. (2023). Polygon Simplification for the Efficient Approximate Analytics of Georeferenced Big Data. Sensors, 23(19), 8178.[available online](https://www.mdpi.com/1424-8220/23/19/8178)
    [^4]: Al Jawarneh, I. M., Bellavista, P., Corradi, A., Foschini, L., & Montanari, R. (2021). QoS-aware approximate query processing for smart cities spatial data streams. Sensors, 21(12), 4160. [available online](https://www.mdpi.com/1424-8220/21/12/4160)
    [^5]: Al Jawarneh, I. M., Bellavista, P., Corradi, A., Foschini, L., & Montanari, R. (2020, September). Spatially representative online Big Data sampling for smart cities. In 2020 IEEE 25th International Workshop on Computer Aided Modeling and Design of Communication Links and Networks (CAMAD) (pp. 1-6). IEEE.[presentation available online](https://isamaljawarneh.github.io/talks/CAMAD20.pdf)
    [^6]: Al Jawarneh, I. M., Bellavista, P., Corradi, A., Foschini, L., & Montanari, R. (2023). SpatialSSJP: QoS-Aware Adaptive Approximate Stream-Static Spatial Join Processor. IEEE Transactions on Parallel and Distributed Systems. [available online](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10309986)
    [^7]: Al Jawarneh, I. M., Foschini, L., & Bellavista, P. (2023). Efficient Integration of Heterogeneous Mobility-Pollution Big Data for Joint Analytics at Scale with QoS Guarantees. Future Internet, 15(8), 263. [available online](https://www.mdpi.com/1999-5903/15/8/263)
    [^8]:Al Jawarneh, I. M., Bellavista, P., Corradi, A., Foschini, L., & Montanari, R. (2021, October). Efficiently integrating mobility and environment data for climate change analytics. In 2021 IEEE 26th International Workshop on Computer Aided Modeling and Design of Communication Links and Networks (CAMAD) (pp. 1-5). IEEE.[presentation available online](https://isamaljawarneh.github.io/talks/CAMAD21.pdf)
    [^9]:Al Jawarneh, I. M., Bellavista, P., Corradi, A., Foschini, L., & Montanari, R. (2020). Efficient QoS-aware spatial join processing for scalable NoSQL storage frameworks. IEEE Transactions on Network and Service Management, 18(2), 2437-2449.[available online](https://isamaljawarneh.github.io/pubs/TNSM3034150.pdf)
    [^10]: Aljawarneh, I. M., Bellavista, P., Corradi, A., Montanari, R., Foschini, L., & Zanotti, A. (2017, July). Efficient spark-based framework for big geospatial data query processing and analysis. In 2017 IEEE symposium on computers and communications (ISCC) (pp. 851-856). IEEE. [available online](https://www.academia.edu/download/55478212/08024633.pdf)
-------------------------

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

