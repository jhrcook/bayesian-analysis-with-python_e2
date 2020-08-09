# Ch 2. Programming Probabilistically

In this chapter, we will cover the following topics:

- Probabilistic programming
- PyMC3 primer
- The coin-flipping problem revisited
- Summarizing the posterior
- The Gaussian and student's t models
- Comparing groups and the effect size
- Hierarchical models and shrinkage


```python
import numpy as np
import pandas as pd
from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
```

## Probabilistic programming

> Bayesian statistics is conceptually very simple; we have the knows and the unknowns; we use Bayes' theorem to condition the latter on the former. If we are lucky, this process will reduce the uncertainty about the unknowns. Generally, we refer to the knowns as data and treat it like a constant, and the unknowns as parameters and treat them as probability distributions.

## PyMC3 primer

- a library for probabilistic programming
- uses NumPy and Theano
    - Theano is a deep learning algorithm that supplies the automatic differentiation required for sampling by PyMC3
    - Theano also compiles the code to C for faster execution
- Theano is no longer developed, but the PyMC devs are currently maintaining it
    - the next version of PyMC will use a different backend

### Flipping coins the PyMC3 way

- make synthetic coin flipping data, but we know the real $\theta$ value


```python
np.random.seed(123)
trials = 4
theta_real = 0.35
data = stats.bernoulli.rvs(p=theta_real, size=trials)
data
```




    array([1, 0, 0, 0])



### Model specification

- need to specify the likelihood function and prior probability distribution
    - likelihood: binomial distribution with $n=1$ and $p=\theta$
    - prior: beta distribution with $\alpha=1$ and $\beta=1$
        - this beta distribution is equivalent to a uniform distirbution from $[0,1]$

$$
\theta \sim \text{Beta}(\alpha, \beta) \\
y \sim \text{Bern}(p=\theta)
$$

- this model using PyMC3:


```python
with pm.Model() as our_first_model:
    θ = pm.Beta('θ', alpha=1.0, beta=1.0)
    y = pm.Bernoulli('y', p=θ, observed=data)
    trace = pm.sample(1000, random_seed=123)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [θ]




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='4000' class='' max='4000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [4000/4000 00:06<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 15 seconds.


### Pushing the inference button

- the last line of the model specification above is "pressing the inference button"
    - asks for 1,000 samples from the posterior distribution

## Summarizing the posterior

- use `plot_trace()` to see the distribution of sampled values for $\theta$ and the MCMC chains


```python
az_trace = az.from_pymc3(trace=trace, model=our_first_model)
p = az.plot_trace(az_trace)
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_11_0.png)


- use `summary()` to get a Pandas data frame describing the posterior


```python
az.summary(az_trace)
```




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
      <th>θ</th>
      <td>0.336</td>
      <td>0.177</td>
      <td>0.027</td>
      <td>0.663</td>
      <td>0.006</td>
      <td>0.004</td>
      <td>806.0</td>
      <td>806.0</td>
      <td>744.0</td>
      <td>800.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



- use `plot_posterior()` to visualize the posterior distribution of $\theta$


```python
p = az.plot_posterior(az_trace)
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_15_0.png)


### Posterior-based descisions

- sometimes need to go further than just descirbing the posterior and actually make a decision

#### ROPE

- **Region Of Practical Equivalence**: a region of the posteior distibution that would be effectively equivalent to a specific value
    - example: a fair coin has $\theta=0.5$ then the ROPE could be $[0.45, 0.55]$

> A ROPE is an arbitrary interval we choose based on background knowledge. Any value inside this interval is assumed to be of practical equivalence.

- comapre the ROPE against the **Highest-Posterior Density (HPD)**
- 3 possible scenarios:
    - the ROPE and HPD do not overalp: the coin is not fair
    - the ROPE contains the entire HPD: the coin is fair
    - the ROPE partially overlaps the HPD: the coin may be fair or unfair
- can include a range for the ROPE in `plot_posterior()`


```python
p = az.plot_posterior(az_trace, rope = [0.45, 0.55])
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_18_0.png)


- can also compare the posterior against a reference value
    - example: 0.5 for a fair coin


```python
az.plot_posterior(az_trace, rope = [0.45, 0.55], ref_val = 0.5)
```




    array([<AxesSubplot:title={'center':'θ'}>], dtype=object)




![png](02_programming-probabilistically_files/02_programming-probabilistically_20_1.png)


#### Loss functions

- loss functoin is a mathematical formalization of a cost/benefit trade-off
    - indicate how different the true and estimated values of a prameter are
- common loss functions:
    - *quadratic*: $(\theta - \hat{\theta})^2$
    - *absolute*: $| \theta - \hat{\theta} |$
    - *0-1*: $I(\theta \neq \hat{\theta})$
- since we don't know the true parameter value, instead try to find the value of $\hat{\theta}$ that minimizes the **expected loss function**
    - expected loss function is the loss function averages over the entire posterior distribution
- example below shows the absolute loss `lossf_a` and quadratic loss `lossf_b`


```python
grid = np.linspace(0, 1, 200)
theta_pos = trace['θ']
lossf_a = [np.mean(abs(i - theta_pos)) for i in grid]
lossf_b = [np.mean((i - theta_pos)**2) for i in grid]

for lossf, c in zip([lossf_a, lossf_b], ['C0', 'C1']):
    mini = np.argmin(lossf)
    plt.plot(grid, lossf, c)
    plt.plot(grid[mini], lossf[mini], 'o', color=c)
    plt.annotate('{:.2f}'.format(grid[mini]),
                 (grid[mini], lossf[mini] + 0.03), 
                 color=c)
    plt.yticks([])
    plt.xlabel(r'$\hat{\theta}$')
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_22_0.png)


## Gaussians all the way down




```python

```
