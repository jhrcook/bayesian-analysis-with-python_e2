# Ch 5. Model Comparison

1. Posterior predictive checks
2. Occam's razor—simplicity and accuracy
3. Overfitting and underfitting
4. Information criteria
5. Bayes factors
6. Regularizing priors

```python
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as gg
import pymc3 as pm
import seaborn as sns
from scipy import stats

%config InlineBackend.figure_format = 'retina'
```

## Posterior predictive checks (PPC)

- a way to evalutate a model in the context of the purpose of the model
- with multiple models, can use PPC to compare them

```python
dummy_data = pd.DataFrame(np.loadtxt("data/dummy.csv"), columns=["x_1", "y"])

dummy_data = dummy_data.assign(
    x_1s=lambda df: (df.x_1 - df.x_1.mean()) / df.x_1.std(),
    x_2s=lambda df: df.x_1s ** 2,
    y_s=lambda df: (df.y - df.y.mean()) / df.y.std(),
)

(
    gg.ggplot(dummy_data, gg.aes("x_1s", "y_s"))
    + gg.geom_point()
    + gg.theme_classic()
    + gg.labs(x="x (scaled)", y="y (scaled)")
)
```

![png](05_model-comparison_files/05_model-comparison_3_0.png)

    <ggplot: (8765699779082)>

- example: compare linear and polynomial data

```python
with pm.Model() as model_l:
    alpha = pm.Normal("alpha", mu=0, sd=1)
    beta = pm.Normal("beta", mu=0, sd=10)
    epsilon = pm.HalfNormal("epsilon", 5)

    mu = alpha + beta * dummy_data["x_1s"]

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=dummy_data["y_s"])
    trace_l = pm.sample(2000)
    postpredcheck_l = pm.sample_posterior_predictive(trace_l, 2000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [epsilon, beta, alpha]

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='6000' class='' max='6000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [6000/6000 00:05<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 14 seconds.
    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:02<00:00]
</div>

```python
az_l = az.from_pymc3(trace=trace_l, model=model_l, posterior_predictive=postpredcheck_l)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable y_pred's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

```python
with pm.Model() as model_p:
    alpha = pm.Normal("alpha", mu=0, sd=1)
    beta = pm.Normal("beta", mu=0, sd=10, shape=2)
    epsilon = pm.HalfNormal("epsilon", 5)

    mu = alpha + pm.math.dot(beta, dummy_data[["x_1s", "x_2s"]].to_numpy().T)

    y_pred = pm.Normal("y_pred", mu=mu, sd=epsilon, observed=dummy_data["y_s"])
    trace_p = pm.sample(2000)
    postpredcheck_p = pm.sample_posterior_predictive(trace_p, 2000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [epsilon, beta, alpha]

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='6000' class='' max='6000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [6000/6000 00:07<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 14 seconds.
    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/pymc3/sampling.py:1707: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample

<div>
    <style>
        /*Turns off some styling*/
        progress {
            /*gets rid of default border in Firefox and Opera.*/
            border: none;
            /*Needs to be in here for Safari polyfill so background images work as expected.*/
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:02<00:00]
</div>

```python
az_p = az.from_pymc3(trace=trace_p, model=model_p, posterior_predictive=postpredcheck_p)
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable y_pred's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.

```python
x_new = np.linspace(dummy_data.x_1s.min(), dummy_data.x_1s.max(), 100)

alpha_l_post = trace_l["alpha"].mean()
beta_l_post = trace_l["beta"].mean(axis=0)
y_l_post = alpha_l_post + beta_l_post * x_new
ppc_l = pd.DataFrame({"x": x_new, "y": y_l_post, "model": "linear"})

alpha_p_post = trace_p["alpha"].mean()
beta_p_post = trace_p["beta"].mean(axis=0)
y_p_post = alpha_p_post + beta_p_post[0] * x_new + beta_p_post[1] * (x_new ** 2)
ppc_p = pd.DataFrame({"x": x_new, "y": y_p_post, "model": "polynomial"})


(
    gg.ggplot(pd.concat([ppc_l, ppc_p]))
    + gg.aes("x", "y")
    + gg.geom_line(gg.aes(color="model"))
    + gg.geom_point(gg.aes(x="x_1s", y="y_s"), data=dummy_data)
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.theme_classic()
    + gg.theme(legend_position=(0.8, 0.6))
    + gg.labs(
        x="x (scaled)",
        y="y (scaled)",
        title="PPC of linear and polynomial models on dummy data",
    )
)
```

![png](05_model-comparison_files/05_model-comparison_9_0.png)

    <ggplot: (8765700585232)>

## Occam's razor – simplicity and accuracy

## Information criteria

- tools for measuring how well the model fits the data given the complexity of the model

### Model comparison with PyMC3

```python
az.waic(az_l)
```

    Computed from 4000 by 33 log-likelihood matrix
    
              Estimate       SE
    elpd_waic   -13.90     2.65
    p_waic        2.46        -

```python
az.waic(az_p)
```

    Computed from 4000 by 33 log-likelihood matrix
    
              Estimate       SE
    elpd_waic    -4.15     2.35
    p_waic        2.71        -

```python
az.compare({"linear_model": az_l, "polynomal_model": az_p})
```

    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/arviz/stats/stats.py:149: UserWarning: 
    The scale is now log by default. Use 'scale' argument or 'stats.ic_scale' rcParam if you rely on a specific value.
    A higher log-score (or a lower deviance) indicates a model with better predictive accuracy.

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>rank</th>
      <th>loo</th>
      <th>p_loo</th>
      <th>d_loo</th>
      <th>weight</th>
      <th>se</th>
      <th>dse</th>
      <th>warning</th>
      <th>loo_scale</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>polynomal_model</th>
      <td>0</td>
      <td>-4.188210</td>
      <td>2.747927</td>
      <td>0.000000</td>
      <td>0.998221</td>
      <td>2.566588</td>
      <td>0.000000</td>
      <td>False</td>
      <td>log</td>
    </tr>
    <tr>
      <th>linear_model</th>
      <td>1</td>
      <td>-13.922071</td>
      <td>2.487777</td>
      <td>9.733861</td>
      <td>0.001779</td>
      <td>2.156212</td>
      <td>2.657495</td>
      <td>False</td>
      <td>log</td>
    </tr>
  </tbody>
</table>
</div>

```python
az.plot_compare(az.compare({"linear_model": az_l, "polynomal_model": az_p}))
```

    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/arviz/stats/stats.py:149: UserWarning: 
    The scale is now log by default. Use 'scale' argument or 'stats.ic_scale' rcParam if you rely on a specific value.
    A higher log-score (or a lower deviance) indicates a model with better predictive accuracy.





    <AxesSubplot:xlabel='Log'>

![png](05_model-comparison_files/05_model-comparison_16_2.png)

## Bayes factors[sic]

- **Bayes factor**

$$
BF = \frac{\Pr(y | M_0)}{\Pr(y | M_1)}
$$

- if $BF > 1$ than model 0 is better at explaining the results than model 1
    - 1-3: weak
    - 3-10: moderate
    - 10-30: strong
    - 30-100: very strong
    - \>100: extreme
- BF are frequently criticized as  hypothesis tested
    - inference based model comparison is generally preferred

### Regularizing priors

- **regurlaization**: adding a bias to reduce a generalization error without affecting the ability of the model to fit
    - common way is to penalize larger parameter values
    - a Bayesian method is to use a tighter prior distribution
    - a Laplace distribution has a peak at 0 and can be used to introduce sparsity

---

```python
%load_ext watermark
%watermark -d -u -v -iv -b -h -m
```

    Last updated: 2021-01-04
    
    Python implementation: CPython
    Python version       : 3.9.1
    IPython version      : 7.19.0
    
    Compiler    : Clang 10.0.0 
    OS          : Darwin
    Release     : 20.1.0
    Machine     : x86_64
    Processor   : i386
    CPU cores   : 4
    Architecture: 64bit
    
    Hostname: JHCookMac.local
    
    Git branch: master
    
    seaborn   : 0.11.1
    pymc3     : 3.9.3
    arviz     : 0.10.0
    numpy     : 1.19.4
    matplotlib: 3.3.3
    plotnine  : 0.7.1
    pandas    : 1.2.0
    scipy     : 1.6.0
