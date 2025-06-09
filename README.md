# timegpy <img src="img/timegpy.png" align="right" width="120" />

Find informative time-average features using genetic programming

## Installation

Coming soon!

## Purpose

`timegpy` (“genetic programming for time-series features”) is a simple
and lightweight Python package for finding informative time-average
‘features’ that can distinguish between classes. A time-series feature
is a summary statistic which returns a scalar for each time series which
summarises some property, such as the value of the autocorrelation
function at lag 1, or the variance of sliding window variances taken
across the time series (see [this
paper](https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2013.0048),
[this
paper](https://www.sciencedirect.com/science/article/pii/S2405471217304386),
and [this book
chapter](https://www.taylorfrancis.com/chapters/edit/10.1201/9781315181080-4/feature-based-time-series-analysis-ben-fulcher)
for more).

Time-average features — quantities of the functional form
$\langle f(x_1, x_2, \dots, x_n)_t \rangle$ such as
$\langle x_{t}x_{t+1} \rangle$ — have shown utility in solving
time-series problems across the sciences but have yet to be
systematically applied to time-series classification problems.
Time-average features are desirable quantities because they are highly
interpretable—for example, a time-average feature of
$\langle x_{t}x_{t+1} \rangle$ represents the average of the product of
values at each time point and the time point one ahead of it. In the
case of $z$-scored data, this represents the autocorrelation function at
lag 1. This interpretability then leads to an intuitive understanding of
why two or more classes might be well distinguished from one another.

Typically, to use time-series features for classification problems,
users have had to compute large numbers of them using software such as
[`hctsa`](https://github.com/benfulcher/hctsa) or
[`theft`](https://github.com/hendersontrent/theft). This resulting time
series $\times$ feature matrix is then used as input to classification
algorithms, such as a linear support vector machine (SVM). However,
extracting large numbers of potentially uninformative features is
computationally expensive, and [recent
work](https://ieeexplore.ieee.org/abstract/document/9679937) highlighted
that many features within feature sets capture similar information about
the time series (i.e., are redundant) meaning that not all need to be
calculated from the outset. Further, large feature spaces make inference
difficult, especially on small datasets.

To address these issues, `timegpy` takes a different approach; instead
implementing an evolutionary algorithm known as [genetic
programming](https://en.wikipedia.org/wiki/Genetic_programming) to
*search* for informative time-average features through the construction
of symbolic mathematical expressions over generations, the parents of
which are selected based on some fitness attribute—just like a
real-world population.

## Development

`timegpy` is still an active work-in-progress, please check back
regularly for updates and/or new functionality.