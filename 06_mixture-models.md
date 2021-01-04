# Ch 6. Mixture Models

1. Finite mixture models
2. Infinite mixture models
3. Continuous mixture models

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
import theano.tensor as tt

gg.theme_set(gg.theme_minimal())
%config InlineBackend.figure_format = 'retina'
```

## Mixture models (MM)

- occurs when the population is a combination of sub-populations
- do not really have to believe there are sub-populations
    - can be used as a trick to add flexibility to the model

## Finite mixture models

- a finite weighted mixture of two or more distributions
- the probability density of the observed data is a weighted sum of the probability density for $K$ subgroups
    - $w_i$: wieght of each subgroup (a.k.a component); the probability of component $i$
    - $\Pr(y|\theta)_i$: distribution of the subgroup $i$

$$
\Pr(y|\theta) = \sum_{i=1}^{K} w_i \Pr_i(y | \theta_i)
$$

- to solve a MM, need to properly assign each data point to one of the components
    - introduce a random variable $z$ that specifies to which component an observation belongs
        - called a *latent* variable
- example using chemical shift data (from protein NMR)

```python
cs = pd.read_csv(Path("data/chemical_shifts_theo_exp.csv"))
cs_exp = cs["exp"]

(
    gg.ggplot(cs, gg.aes("exp"))
    + gg.geom_histogram(
        gg.aes(y="..density.."), bins=30, alpha=0.3, fill="blue", color="blue"
    )
    + gg.geom_density(color="blue", size=1.5)
    + gg.scale_x_continuous(expand=(0, 0))
    + gg.scale_y_continuous(expand=(0, 0, 0.02, 0))
    + gg.labs(x="exp", y = "density", title="Chemical shift mixture")
)
```

![png](06_mixture-models_files/06_mixture-models_4_0.png)

    <ggplot: (8762545097488)>

### The categorical distribution

- most general discrete distribution
- parameterized using a parameter specifying the probabilities of each possible outcome

```python
example_cat_data = pd.DataFrame(
    {
        "cat": ["A", "A", "A", "B", "B", "B", "B"],
        "y": [0.1, 0.6, 0.3, 0.3, 0.1, 0.1, 0.5],
        "x": [0, 1, 2, 0, 1, 2, 3],
    }
)

(
    gg.ggplot(example_cat_data, gg.aes(x="x", y="y", color="cat"))
    + gg.geom_line(alpha=0.5, size=1)
    + gg.geom_point(size=3)
    + gg.scale_color_brewer(type="qual", palette="Set1")
    + gg.labs(x="x", y="f(x)", title="Example categorial distribution")
)
```

![png](06_mixture-models_files/06_mixture-models_6_0.png)

    <ggplot: (8762545097500)>

### The Dirichlet distribution

- the Dirichlet distribution:
    - lives in the *simplex* (an $n$-dim. triangle)
    - output of the dist. is a $K$-length vector $\alpha$
    - output are restricted to be nonnegative and sum to 1
    - generalization of the beta distribution
        - i.e. extend the beta dist. beyond 2 outcomes
        - the beta dist. has two outcomes and returns a 2-element vector: $(p, 1-p)$
        - for 3 outcomes, need a 3-element vector $(p, q, r)$ where $p+q+r=1$
        - for $K$ outcomes, declare a vector $\alpha$ where $||\alpha|| = K$

![](assets/ch06/Dirichlet-triangles.png)

- in a model, use the Dirichlet dist. as a $K$-sided coin flip model on top of a Gaussian estimated model
    - in the diagram, there are $K$ copies of the model in the rounded-rect
    - $\mu_k$ is dependent upon the group $k$
    - $\sigma_\mu$ and $\sigma_\sigma$ are shared for all groups

![](assets/ch06/Dirichlet-prior-mixture-model.png)

```python
# NOTE: We expect this  model to fit poorly.
# See below for an example of how to reparameterize for a more
# effective fitting process.

clusters = 2

with pm.Model() as model_kg:
    p = pm.Dirichlet("p", a=np.ones(clusters))
    z = pm.Categorical("z", p=p, shape=len(cs.exp))  # group assignment; one per data point
    means = pm.Normal("means", mu=cs.exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal("sd", sd=10)

    y = pm.Normal("y", mu=means[z], sd=sd, observed=cs.exp)
    # trace_kg = pm.sample()  # NOT RUN BECAUSE TOOOOOOO SLOW
```

- the above model fits poorly because the latent variable `z` is explcitly included
- reparameterize:
    - in a MM, the observed vairable $y$ is conditional on the latent variable $z$, that is $\Pr(y|z, \theta)$
    - can consider $z$ is nuisance variable and marginalize to get $\Pr(y | \theta)$
    - can do this in PyMC3 using the `NormalMixture()` distribution to get a Gaussian mixture model

```python
clusters = 2

with pm.Model() as model_mg:
    p = pm.Dirichlet("p", a=np.ones(clusters))
    means = pm.Normal("means", mu=cs.exp.mean(), sd=10, shape=clusters)
    sd = pm.HalfNormal("sd", sd=10)

    y = pm.NormalMixture("y", w=p, mu=means, sd=sd, observed=cs.exp)

    trace_mg = pm.sample(random_seed=123)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sd, means, p]

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
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 00:18<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 26 seconds.
    The rhat statistic is larger than 1.4 for some parameters. The sampler did not converge.
    The estimated number of effective samples is smaller than 200 for some parameters.

```python
az_trace_mg = az.from_pymc3(trace_mg, model=model_mg)
```

```python
az.plot_trace(az_trace_mg, var_names=["means", "p"])
plt.show()
```

![png](06_mixture-models_files/06_mixture-models_12_0.png)

```python
az.summary(trace_mg, var_names=["means", "p"])
```

    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/arviz/data/io_pymc3.py:87: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.

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
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>means[0]</th>
      <td>52.154</td>
      <td>5.329</td>
      <td>46.242</td>
      <td>57.681</td>
      <td>3.747</td>
      <td>3.169</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>63.0</td>
      <td>1.83</td>
    </tr>
    <tr>
      <th>means[1]</th>
      <td>52.174</td>
      <td>5.299</td>
      <td>46.358</td>
      <td>57.664</td>
      <td>3.726</td>
      <td>3.152</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>61.0</td>
      <td>1.83</td>
    </tr>
    <tr>
      <th>p[0]</th>
      <td>0.500</td>
      <td>0.410</td>
      <td>0.076</td>
      <td>0.924</td>
      <td>0.288</td>
      <td>0.244</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>93.0</td>
      <td>1.83</td>
    </tr>
    <tr>
      <th>p[1]</th>
      <td>0.500</td>
      <td>0.410</td>
      <td>0.076</td>
      <td>0.924</td>
      <td>0.288</td>
      <td>0.244</td>
      <td>2.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>93.0</td>
      <td>1.83</td>
    </tr>
  </tbody>
</table>
</div>

### Non-identifiability of mixture models

- looking at the trace plots and summary table above, notive that the distributions for *both* means and both $p$ are bimodal and that the average values for these parameters are the same
    - caused by **parameter non-identifiability** because can make multiple different models where one mean is the either mode and the other is the remaining mode (and the same for $p$)
    - in mixture models, also known as **lbel-switching problem**
    - two ways to prevent non-identifiability in MM:
        1. force the component sto be ordered (e.g. arrange the means of the components in increasing order)
        2. use informative priors
- in PyMC3, can add `pm.potential()` to the likelihood of the model
    - is an arbitrary factor that does not necessarily depend on the data
    - thus, does not add any parameters to the model
    - example of adding a constraing to a model:
        - if the constraint is not violation, add 0 to the likelihood
        - if is violated, add $-\infty$ to make the outcome impossible
        - thus, the model considers parameters that violate the constraints to be impossible without affecting the rest of the values

```python
clusters = 2

with pm.Model() as model_mgp:
    p = pm.Dirichlet("p", a=np.ones(clusters))
    means = pm.Normal(
        "means",
        mu=np.array([0.9, 1]) * cs_exp.mean(),
        sd=10,
        shape=clusters
    )
    sd = pm.HalfNormal("sd", sd=10)
    order_means = pm.Potential(
        "order_means",
        tt.switch(means[1]-means[0] < 0, -np.inf, 0)
    )
    y = pm.NormalMixture("y", w=p, mu=means, sd=sd, observed=cs_exp)
    trace_mgp = pm.sample(1000, random_seed=123)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [sd, means, p]

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
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 00:20<00:00 Sampling 2 chains, 0 divergences]
</div>

    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 27 seconds.

```python
varnames = ['means', 'p']
az.plot_trace(trace_mgp, varnames)
plt.show()
```

    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/arviz/data/io_pymc3.py:87: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.

![png](06_mixture-models_files/06_mixture-models_16_1.png)

```python
az.summary(trace_mgp)
```

    /usr/local/Caskroom/miniconda/base/envs/bayesian-analysis-with-python_e2/lib/python3.9/site-packages/arviz/data/io_pymc3.py:87: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.

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
      <th>mean</th>
      <th>sd</th>
      <th>hdi_3%</th>
      <th>hdi_97%</th>
      <th>mcse_mean</th>
      <th>mcse_sd</th>
      <th>ess_mean</th>
      <th>ess_sd</th>
      <th>ess_bulk</th>
      <th>ess_tail</th>
      <th>r_hat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>means[0]</th>
      <td>46.811</td>
      <td>0.438</td>
      <td>46.040</td>
      <td>47.687</td>
      <td>0.009</td>
      <td>0.007</td>
      <td>2185.0</td>
      <td>2185.0</td>
      <td>2188.0</td>
      <td>1432.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>means[1]</th>
      <td>57.461</td>
      <td>0.098</td>
      <td>57.290</td>
      <td>57.650</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2255.0</td>
      <td>2255.0</td>
      <td>2253.0</td>
      <td>1686.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>p[0]</th>
      <td>0.090</td>
      <td>0.009</td>
      <td>0.072</td>
      <td>0.106</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1818.0</td>
      <td>1818.0</td>
      <td>1804.0</td>
      <td>1639.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>p[1]</th>
      <td>0.910</td>
      <td>0.009</td>
      <td>0.894</td>
      <td>0.928</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1818.0</td>
      <td>1817.0</td>
      <td>1804.0</td>
      <td>1639.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>sd</th>
      <td>3.651</td>
      <td>0.074</td>
      <td>3.506</td>
      <td>3.788</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>2236.0</td>
      <td>2222.0</td>
      <td>2267.0</td>
      <td>1619.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>

- can also add a constraint to ensure that each component has a non-null probability
    - i.e. that each component gets at least one observation

```python
min_p = 0.01
p_min = pm.Potential("p_min", tt.switch(tt.min(p) < min_p, -np.inf, 0))
```

### How to choose K

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

```python

```

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
    
    pandas    : 1.2.0
    arviz     : 0.10.0
    matplotlib: 3.3.3
    scipy     : 1.6.0
    numpy     : 1.19.4
    plotnine  : 0.7.1
    pymc3     : 3.9.3
    theano    : 1.0.5
    seaborn   : 0.11.1
