Quickstart
==========

Installation
------------

You can install ``timegpy`` via GitHub:

.. code::
   
     pip install git+https://github.com/hendersontrent/timegpy.git

Usage
-----

``timegpy`` ("genetic programming for time-average features") is a Python package for finding informative time-average 'features' that can distinguish between classes. A time-series feature is a summary statistic which returns a scalar for each time series which summarises some property, such as the value of the autocorrelation function at lag 1, or the variance of sliding window variances taken across the time series (see `this paper <https://royalsocietypublishing.org/doi/abs/10.1098/rsif.2013.0048>`_, `this paper <https://www.sciencedirect.com/science/article/pii/S2405471217304386>`_, and `this book chapter <https://www.taylorfrancis.com/chapters/edit/10.1201/9781315181080-4/feature-based-time-series-analysis-ben-fulcher>`_ for more). 

Time-average features---quantities of the functional form :math:`$\langle f(x_1, x_2, \dots, x_n)_t \rangle$` such as :math:`$\langle x_{t}x_{t+1} \rangle$`---have shown utility in solving time-series problems across the sciences but have yet to be systematically applied to time-series classification problems. Time-average features are highly interpretable, which means they can be used to develop an intuitive understanding of why two classes might be well distinguished from one another by their temporal dynamics.

This tutorial will walk through basic functionality of the package using a simulated example. We will first generate some data, where we have :math:`n = 100` samples from an `autoregressive process of lag 1 <https://en.wikipedia.org/wiki/Autoregressive_model>`_ with an autoregressive coefficient of :math:`\phi = 0.8`, and :math:`n = 100` samples drawn from simple Gaussian noise (with mean 0 and standard deviation 1), where every time series is :math:`T = 100` long:

.. code::
   
   >>> import numpy as np
   >>> from timegpy.gp import tsgp
   >>> from timegpy.plots import plot_feature

   >>> # Simulate Gaussian noise and AR(1) data

   >>> def generate_ar1_vs_noise(N, T, phi, seed=None):
   >>>      if seed is not None:
   >>>          np.random.seed(seed)

   >>>      # AR(1) samples

   >>>      ar1_data = np.zeros((N, T))
   >>>      for i in range(N):
   >>>          noise = np.random.normal(0, 1, T)
   >>>          ar1 = np.zeros(T)
   >>>          ar1[0] = noise[0]
   >>>          for t in range(1, T):
   >>>              ar1[t] = phi * ar1[t - 1] + noise[t]
   >>>          ar1_data[i] = ar1

   >>>      # Gaussian noise samples

   >>>      noise_data = np.random.normal(0, 1, (N, T))

   >>>      # Combine

   >>>      X = np.vstack([ar1_data, noise_data])

   >>>      # Class labels

   >>>      y = np.array([1] * N + [0] * N)

   >>>      return X, y

   >>> X, y = generate_ar1_vs_noise(N=100, T=100, phi=0.8, seed=123)

Structure of time-average feature expressions in timegpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In ``timegpy``, features are represented as strings to the user, but trees internally. For example, a time-average feature representing lag 1 autocorrelation function (on *z*-scored data) which is mathematically written as :math:`$\langle x_{t}x_{t+1} \rangle$` would be represented in ``timegpy`` as ``"X_t+0 * X_t+1"`` or, more correctly, ``"mean(X_t+0 * X_t+1)"``. More complex features may include exponents, such as ``"X_t+0 + X_t+1 ^ 3"`` and/or numerous other combinations of time lags.

From a statistical perspective, for this tutorial example, we would expect to see the *best* performing feature to be ``"X_t+0 * X_t+1"`` as this corresponds to the value of the autocorrelation function at lag 1---which we know from the data simulation code above to be the distinguishing temporal difference between the two processes. This creates a nice ground truth test case for the algorithm.

Doing genetic programming in timegpy
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The core function in ``timegpy`` is ``tsgp`` ('time-series genetic programming'). We can exercise high degrees of control over the algorithm's evolution by adjusting the large number of available arguments. Here is a simple call using all of the default parameters:

.. code::
   
   >>> df_all, df_best = tsgp(X, y)

``tsgp`` returns two objects:

1. Data frame containing all time-average features across all generations and their fitness scores
2. Data frame containing the best individual time-average feature and its fitness score

Despite the simplicity of the above call, it is highly likely that users will seek to adjust the numerous parameters to their task. Here is a breakdown of the available arguments:

* ``X`` (array): ID by time matrix containing time-series data
* ``y`` (array): vector of class labels for each row of ``X``
* ``pop_size`` (int): size of each population. Defaults to ``100``
* ``n_generations`` (int): maximum number of generations. Defaults to ``5``
* ``fitness_threshold`` (float): objective function value which if equalled or exceeded, will terminate the algorithm. Defaults to ``0.95``
* ``p_point_mutation`` (float): probability of point mutation occurring. Defaults to ``0.01``
* ``p_subtree_mutation`` (float): probability of subtree mutation occurring. Defaults to ``0.01``
* ``p_hoist_mutation`` (float): probability of hoist mutation occurring. Defaults to ``0.01``
* ``p_crossover`` (float): probability of crossover occurring. Defaults to ``0.9``
* ``p_exponent`` (float): probability of a time lag being exponentiated. Defaults to ``0.3``
* ``tournament_size`` (int): size of each tournament to find a suitable parent. Defaults to ``20``
* ``use_parsimony`` (bool): whether to use parsimony-adjusted fitness instead of raw fitness. Defaults to ``True``
* ``auto_parsimony`` (bool): whether to calculate generational parsimony coefficients dynamically. Defaults to ``True``
* ``parsimony_coefficient`` (float): if ``auto_parsimony = False``, this static coefficient for parsimony will be applied to all generations. Defaults to ``0.001``
* ``verbose`` (bool): whether to print updates of algorithm progress. Defaults to ``False``
* ``max_depth`` (int): maximum number of time-lag terms allowed in a single feature expression. Defaults to ``8``
* ``max_lag`` (int): maximum time-lag allowed in a single feature expression. Defaults to ``8``
* ``max_exponent`` (int): maximum exponent allowed. Defaults to ``5``
* ``seed`` (int): fixes Python's random seed for reproducibility. Defaults to ``123``
* ``n_generation_improve`` (int): number of generations of no fitness improvement before algorithm terminates early. Defaults to ``1``
* ``z_score`` (bool): whether to z-score input data X. Defaults to ``True``
* ``n_procs`` (int): number of processes to use if parallel processing is desired. Defaults to ``1`` for serial processing

Important parameter notes
^^^^^^^^^^^^^^^^^^^^^^^^^

``fitness_threshold`` must be :math:`0 \geq \text{fitness\_threshold} \leq 1` as the current objective function maximises values between :math:`0` and :math:`1`.

The values of ``p_point_mutation``, ``p_subtree_mutation``, ``p_hoist_mutation``, and ``p_crossover`` must sum to :math:`\textless 1` as the remaining probability is allocated to 'no change'.

``parsimony_coefficient``, if used, must be :math:`\textless 1` otherwise it does not represent a complexity penalty.

Additional graphical tools
^^^^^^^^^^^^^^^^^^^^^^^^^^

``timegpy`` also contains functionality for interpreting and visualising genetic programming outputs. For example, users may seek to visualise class separation according to the best time-average feature (or any other). The convenience function ``plot_hist`` has been included for this purpose. It only requires a time-average feature expression as a string (using the conventions of ``timegpy``), the input data ``X``, and the class label vector ``y``. Here is an example using the best found expression from the above example:

.. code::
   
   >>> expression = df_best.iloc[0]['expression']
   >>> plot_hist(expression, X, y, z_score=True)

.. image:: images/ar1-plot.png
  :width: 600
  :alt: Noise vs AR(1) histogram on the best individual feature.

Intuitively, we see the Gaussian noise time series distributed around a feature value of :math:`0` and the AR(1) data (Class 1) distributed around :math:`0.8`---which we know to be the autoregressive coefficient we used to generate the data. This, combined with the fact that ``"X_t+0 * X_t+1"`` was found to be the best time-average feature for classifying the time series, solidifies that the algorithm is working as expected.

There is also the ability to plot the Pareto front of all features found across all generations:

.. code::
   
   >>> plot_pareto(df_all, df_best, use_parsimony=True, level=0.95)

.. image:: images/pareto-front.png
  :width: 600
  :alt: Pareto front of all feature program sizes and adjusted fitness values.

Evaluating individual time-average feature expressions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Outside of the core genetic programming algorithm contained in ``tsgp``, ``timegpy`` can also calculate time-average feature values for any given string representation of an expression and the input matrix:

.. code::
   
   >>> feature_values = evaluate_expression("mean((X_t+0 * X_t+1))", X, z_score=True)

Multiclass problems
^^^^^^^^^^^^^^^^^^^

Since the current fitness metric is an (adjusted) :math:`\eta^{2}` from an ANOVA (which can have multiple groups), there are no additional requirements for multiclass problems. Let's generate a three-class problem of Gaussian noise versus AR(1) process versus AR(2) process and run ``tsgp``:

.. code::
   
   >>> def generate_noise_vs_ar1_vs_ar2(N, T, phi1=0.8, phi2=0.5, phi3=0.3, seed=None):
   >>>  if seed is not None:
   >>>      np.random.seed(seed)

   >>>  # AR(1) samples

   >>>  ar1_data = np.zeros((N, T))
   >>>  for i in range(N):
   >>>      noise = np.random.normal(0, 1, T)
   >>>      ar1 = np.zeros(T)
   >>>      ar1[0] = noise[0]
   >>>      for t in range(1, T):
   >>>          ar1[t] = phi1 * ar1[t - 1] + noise[t]
   >>>      ar1_data[i] = ar1

   >>>  # AR(2) samples

   >>>  ar2_data = np.zeros((N, T))
   >>>  for i in range(N):
   >>>      noise = np.random.normal(0, 1, T)
   >>>      ar2 = np.zeros(T)
   >>>      ar2[0] = noise[0]
   >>>      ar2[1] = noise[1]
   >>>      for t in range(2, T):
   >>>          ar2[t] = phi2 * ar2[t - 1] + phi3 * ar2[t - 2] + noise[t]
   >>>      ar2_data[i] = ar2

   >>>  # Gaussian noise samples

   >>>  noise_data = np.random.normal(0, 1, (N, T))

   >>>  # Combine and label

   >>>  X = np.vstack([noise_data, ar1_data, ar2_data])
   >>>  y = np.array([0] * N + [1] * N + [2] * N)

   >>>  return X, y

   >>> X2, y2 = generate_noise_vs_ar1_vs_ar2(N=100, T=100, phi1=0.8, phi2=0.5, phi3=0.3, seed=123)

   >>> X2, y2, df_all2, df_best2 = tsgp(X2, y2)

We can now easily visualise the best performing feature and how each class is distributed on it:

.. code::
   
   >>> expression2 = df_best2.iloc[0]['expression']
   >>> plot_hist(expression2, X2, y2, z_score=True)

.. image:: images/noise-ar1-ar2.png
  :width: 600
  :alt: Noise vs AR(1) vs AR(2) histogram on the best individual feature.

Additional functionality
^^^^^^^^^^^^^^^^^^^^^^^^

``timegpy`` also contains a host of other functionality, such as the function ``feature_tree`` which prints ASCII-style tree representations of time-average features to the console:

.. code::
   
   >>> feature_tree("mean((X_t * X_t+1^3))")
   >>>
   >>> └── *
   >>>  ├── X_t
   >>>  └── ^
   >>>      ├── X_t+1
   >>>      └── 3
