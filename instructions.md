# instructions
## some insights:
---------------------------------------
<!-- Task 2 -->
# [ ] Task 2! 
# `Update: April 6, 2024`
### `N.B.` references are available in the end of this instruction file!
# `Required OPTIMIZATION ==> IMPORTANT!`
## `TODO:` 
- [ ] Using the algorithms that you have developed, (being it GMM, KDE, MDN), 
generate sufficient amount of new `data samples`. The size of the `generated` data should be the same as the size of the `original` data that has been used for training for generating the data. You should generate sufficient amount of `new samples` to be able to perform subsequent tests. For example, generate 100K (one hundred thousands) new samples. Thereafter, you need to test the accuracy and the quality of the generated data based on several query workloads as follows (**N.B.** first, for the NYC Air Quality (AQ) data, the target variable is `pm25`)
    A) geo-statistics queries (e.g., `count` and `average`). Example query is `"what is the average pm25 each neighborhood of NYC"`, OR `"what is the pm25 readings count in each neighborhood in NYC"`. 
        - this 
--------------------------------------
- Have a look at the Example: GMM for Generating New Data [Gaussian Mixture Models](https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html). AND
- Kernel Density Estimation [KDE](https://jakevdp.github.io/PythonDataScienceHandbook/05.13-kernel-density-estimation.html)
- the data file contains granular (meter-meter level) low-cost air quality data, measuring most importantly Particulate Matters (PM10 and PM2.5), in addition to some other features including location represented as longitudes/latitudes!
