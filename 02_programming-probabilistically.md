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
import seaborn as sns
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



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 16 seconds.


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



### Gaussian inferences

- the data need not be a Gaussian, just that a Gaussian is a *reasonable approximation* of the data
    - some example data of NMR chemical shifts


```python
data = np.loadtxt('data/chemical_shifts.csv')
data
```




    array([51.06, 55.12, 53.73, 50.24, 52.05, 56.4 , 48.45, 52.34, 55.65,
           51.49, 51.86, 63.43, 53.  , 56.09, 51.93, 52.31, 52.33, 57.48,
           57.44, 55.14, 53.93, 54.62, 56.09, 68.58, 51.36, 55.47, 50.73,
           51.94, 54.95, 50.39, 52.91, 51.5 , 52.68, 47.72, 49.73, 51.82,
           54.99, 52.84, 53.19, 54.52, 51.46, 53.73, 51.61, 49.81, 52.42,
           54.3 , 53.84, 53.16])




```python
az.plot_kde(data, rug=True)
plt.yticks([0], alpha=0)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_26_0.png)


- we do not know the mean nor standard deviation, so set priors for them
- example model

$$
\mu \sim U(l,h) \\
\sigma \sim | \mathcal{N}(0, \sigma_{\sigma}) | \\
y \sim \mathcal{N}(\mu, \sigma)
$$

- this model in code


```python
with pm.Model() as model_g:
    µ = pm.Uniform('µ', lower=40, upper=70)
    σ = pm.HalfNormal('σ', sd=10)
    y = pm.Normal('y', mu=µ, sd=σ, observed=data)
    trace_g = pm.sample(1000)

az_trace_g = az.from_pymc3(trace=trace_g, model=model_g)
az.plot_trace(az_trace_g)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [σ, µ]




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



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 12 seconds.





    array([[<AxesSubplot:title={'center':'µ'}>,
            <AxesSubplot:title={'center':'µ'}>],
           [<AxesSubplot:title={'center':'σ'}>,
            <AxesSubplot:title={'center':'σ'}>]], dtype=object)




![png](02_programming-probabilistically_files/02_programming-probabilistically_28_4.png)



```python
az.plot_pair(az_trace_g, kind='kde', fill_last=False)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_29_0.png)


- can get access to values in a `trace` object by indexing with the parameter name
    - returns a NumPy array


```python
len(trace_g['σ'])
```




    2000




```python
az.summary(az_trace_g)
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
      <th>µ</th>
      <td>53.503</td>
      <td>0.512</td>
      <td>52.608</td>
      <td>54.518</td>
      <td>0.013</td>
      <td>0.009</td>
      <td>1490.0</td>
      <td>1486.0</td>
      <td>1504.0</td>
      <td>1241.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>3.533</td>
      <td>0.377</td>
      <td>2.874</td>
      <td>4.258</td>
      <td>0.009</td>
      <td>0.006</td>
      <td>1731.0</td>
      <td>1715.0</td>
      <td>1741.0</td>
      <td>1297.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



- use the `sample_posterior_predictive()` function to do **posterior predictive checks**


```python
y_pred_g = pm.sample_posterior_predictive(trace_g, 100, model_g)
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/pymc3/sampling.py:1617: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  <progress value='100' class='' max='100' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [100/100 00:00<00:00]
</div>




```python
data_ppc = az.from_pymc3(trace=trace_g, model=model_g, posterior_predictive=y_pred_g)
ax = az.plot_ppc(data_ppc, figsize=(12, 6), mean=False)
plt.legend(fontsize=15)
plt.show()
```

    arviz.data.io_pymc3 - WARNING - posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



![png](02_programming-probabilistically_files/02_programming-probabilistically_35_1.png)


- from the plot, see that the mean is shifted to the right and variance is quite high
    - see a better model for this data in the next section

### Robust inferences

#### Student's t-distribution

- instead of removing outliers, replace the Gaussian with a Student's t-distribution
    - has 3 parameters: mean, scale, and degrees of freedom ($\nu \in [0, \infty]$)
    - call the $\nu$ the *normality parameter* because it controls how "normal-like" the distribution is
        - $\nu = 1$: heavy tails (Cauchy or Lorentz distributions)
        - as $\nu$ gets larger, it appraoches a Gaussian
    - for $\nu \leq 1$: no defined mean
    - for $\nu \leq 2$: no defined variance (or std. dev.)


```python
plt.figure(figsize=(10, 6))
x_values = np.linspace(-10, 10, 500)
for dof in [1, 2, 30]:
    distrib = stats.t(dof)
    x_pdf = distrib.pdf(x_values)
    plt.plot(x_values, x_pdf, label=fr'$\nu = {dof}$', lw=3)

x_pdf = stats.norm.pdf(x_values)
plt.plot(x_values, x_pdf, 'k--', label=r'$\nu = \infty$')
plt.xlabel('x')
plt.yticks([])
plt.legend()
plt.xlim(-5, 5)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_39_0.png)


- rewrite the previous model with Student's t-distribution instead of a Gaussian
    - we will provide $\nu$ with an unimformative exponential prior to indicate we think it will be mostly normal

$$
\mu \sim \mathcal{U}(l,h) \\
\sigma \sim | \mathcal{N}(0, \sigma_{\sigma}) |  \\
\nu \sim Exp(\lambda) \\
y \sim \mathcal{T}(\mu, \sigma, \nu)
$$


```python
with pm.Model() as model_t:
    µ = pm.Uniform('µ', 40, 75)
    σ = pm.HalfNormal('σ', sd=10)
    ν = pm.Exponential('ν', 1/30)  # d.o.f. is parameterized as the inverse mean
    y = pm.StudentT('y', mu=µ, sd=σ, nu=ν, observed=data)
    trace_t = pm.sample(1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [ν, σ, µ]




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
  100.00% [4000/4000 00:08<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 15 seconds.



```python
az_trace_t = az.from_pymc3(trace=trace_t, model=model_t)
az.plot_trace(az_trace_t)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_42_0.png)



```python
az.summary(az_trace_t)
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
      <th>µ</th>
      <td>53.018</td>
      <td>0.388</td>
      <td>52.317</td>
      <td>53.736</td>
      <td>0.012</td>
      <td>0.008</td>
      <td>1124.0</td>
      <td>1124.0</td>
      <td>1124.0</td>
      <td>1257.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>σ</th>
      <td>2.199</td>
      <td>0.415</td>
      <td>1.426</td>
      <td>2.950</td>
      <td>0.014</td>
      <td>0.010</td>
      <td>844.0</td>
      <td>844.0</td>
      <td>838.0</td>
      <td>823.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ν</th>
      <td>4.854</td>
      <td>5.626</td>
      <td>1.386</td>
      <td>10.063</td>
      <td>0.208</td>
      <td>0.147</td>
      <td>729.0</td>
      <td>729.0</td>
      <td>854.0</td>
      <td>904.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
y_ppc_t = pm.sample_posterior_predictive(trace_t, 100, model_t, random_seed=123)
y_pred_t = az.from_pymc3(trace=trace_t, model=model_t, posterior_predictive=y_ppc_t)
ax = az.plot_ppc(y_pred_t, figsize=(12, 6), mean=False)
plt.legend(fontsize=15)
plt.xlim(40, 70)
plt.show()
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/pymc3/sampling.py:1617: UserWarning: samples parameter is smaller than nchains times ndraws, some draws and/or chains may not be represented in the returned posterior predictive sample
      warnings.warn(




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
  <progress value='100' class='' max='100' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [100/100 00:00<00:00]
</div>



    arviz.data.io_pymc3 - WARNING - posterior predictive variable y's shape not compatible with number of chains and draws. This can mean that some draws or even whole chains are not represented.



![png](02_programming-probabilistically_files/02_programming-probabilistically_44_3.png)


## Groups comparison

- this author emphasizes useing effect size and uncertainty over just presenting a p-value (or some other yes/no indicator)
- specifically, 3 tools:
    1. a posterior plot with a reference value
    2. the Cohen's d
    3. the probability of superiority

### Cohen's d

formula: $\frac{\mu_2 - \mu_1}{\sqrt{\frac{\sigma_2^2 + \sigma_1^2}{2}}}$

- the difference in means with respect to the pooled standard deviation of both groups

### Probability of superiority

- the probability that a data point taken at random from one group has a larger value than one taken at random from the other group
- if the data is normally distributed, can compute using Cohen's d ($\delta$) and the cumulative normal distribution $\Phi$

$ps = \Phi(\frac{\delta}{\sqrt{2}})$

- it can also be computed if we have direct samples from the posterior (such as from MCMC)

### The tips dataset

- example: study the effect of the day of the week on the amount of tips at a restaurant


```python
tips = pd.read_csv("data/tips.csv")
tips.head()
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
      <th>total_bill</th>
      <th>tip</th>
      <th>sex</th>
      <th>smoker</th>
      <th>day</th>
      <th>time</th>
      <th>size</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>16.99</td>
      <td>1.01</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10.34</td>
      <td>1.66</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>21.01</td>
      <td>3.50</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>23.68</td>
      <td>3.31</td>
      <td>Male</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>24.59</td>
      <td>3.61</td>
      <td>Female</td>
      <td>No</td>
      <td>Sun</td>
      <td>Dinner</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
plt.figure(figsize=(8, 5))
sns.violinplot(x='day', y='tip', data=tips)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_50_0.png)


- make some variables to make handling the data easier


```python
tip = tips['tip'].values
idx = pd.Categorical(tips['day'], categories=['Thur', 'Fri', 'Sat', 'Sun']).codes
groups = len(np.unique(idx))
```

- same model as `model_g` except now $\mu$ and $\sigma$ are vectors, one value per day of the week


```python
with pm.Model() as comparing_groups:
    µ = pm.Normal('µ', mu=0, sd=10, shape=groups)
    σ = pm.HalfNormal('σ', sd=10, shape=groups)
    
    y = pm.Normal('y', mu=µ[idx], sd=σ[idx], observed=tip)
    
    trace_cg = pm.sample(5000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [σ, µ]




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
  <progress value='12000' class='' max='12000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [12000/12000 00:20<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 5_000 draw iterations (2_000 + 10_000 draws total) took 28 seconds.



```python
az_trace_cg = az.from_pymc3(trace=trace_cg, model=comparing_groups)
az.plot_trace(az_trace_cg)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_55_0.png)



```python
dist = stats.norm()

fig, ax = plt.subplots(3, 2, figsize=(14, 8), constrained_layout=True)

comparisons = [(i, j) for i in range(4) for j in range(i+1, 4)]
pos = [(k, l) for k in range(3) for l in (0, 1)]

for (i, j), (k, l) in zip(comparisons, pos):
    means_diff = trace_cg['µ'][:, i] - trace_cg['µ'][:, j]
    d_cohen = (means_diff / np.sqrt((trace_cg['σ'][:, i]**2 + trace_cg['σ'][:, j]**2) / 2)).mean()
    ps = dist.cdf(d_cohen / (2**0.5))
    
    az.plot_posterior(means_diff, ref_val=0, ax=ax[k, l])
    ax[k, l].set_title(f'$\mu_{i}-\mu_{j}$')
    ax[k, l].plot(
        0, label=f'Cohen\'s d = {d_cohen:.2f}\nProb sup = {ps:.2f}', alpha=0
    )
    ax[k,l].legend()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_56_0.png)


## Hierarchical models

- instead of fixing parameters of the priors to some constant, estimate the priors over all of the data
    - called *hyper-priors* with *hyperparameters*
- example of mock water quality measurements collected from 3 regions within a city
    - values of lead above healthy standards are marked with 0 and those below are makred with 1


```python
N_samples = [30, 30, 30]
G_samples = [18, 18, 18]

group_idx = np.repeat(np.arange(len(N_samples)), N_samples)
data = []
for i in range(0, len(N_samples)):
    data.extend(np.repeat([1, 0], [G_samples[i], N_samples[i]-G_samples[i]]))

print(group_idx[:20])
print(data[:20])
print(len(data))
```

    [0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0]
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
    90


- model is a the same as for a coin flip except:
    - defined 2 hhyper-priors that will influence the beta prior
    - instead of putting hyper-priors on the parameters $\alpha$ and $\beta$, indirectly define them with $\mu$ (mean of the beta distribution)and $\kappa$ (the precision of the beta distribution; effectively the inverse of std. dev.)

$$
\mu \sim \text{Beta}(\alpha_{\mu}, \beta_{\mu}) \\
\kappa \sim | \mathcal{N}(0, \sigma_\kappa) | \\
\alpha = \mu \times \kappa \\
\beta = (1 - \mu) \times \kappa \\
\theta_i \sim \text{Beta}(\alpha_i, \beta_i) \\
y_i \sim \text{Bernoulli}(\theta_i)
$$


```python
with pm.Model() as model_h:
    µ = pm.Beta('µ', 1.0, 1.0)
    κ = pm.HalfNormal('κ', 10)
    
    θ = pm.Beta('θ', alpha=µ*κ, beta=(1.0-µ)*κ, shape=len(N_samples))
    y = pm.Bernoulli('y', p=θ[group_idx], observed=data)
    
    trace_h = pm.sample(2000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [θ, κ, µ]




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
  <progress value='6000' class='' max='6000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [6000/6000 00:11<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 19 seconds.



```python
az_trace_h = az.from_pymc3(trace=trace_h, model=model_h)
az.plot_trace(az_trace_h)
plt.show()
```


![png](02_programming-probabilistically_files/02_programming-probabilistically_61_0.png)


### Shrinkage

- when the parameters share information through the hyper-prior, they *shrink* towards the value instead of remaining completely separate
    - paritally pooling the data
- shrinkinage contributes to more stable inferences
- can use a more informative prior on the hyper-prior to increase the amount of shrinkage

### One more example

- an example of a hierarchical model with protein NMR data
    - want to compare the differences between theoretical (`theo`) and experimental (`exp`) chemical shift values


```python
cs_data = pd.read_csv('data/chemical_shifts_theo_exp.csv')
cs_data.head()
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
      <th>ID</th>
      <th>aa</th>
      <th>theo</th>
      <th>exp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1BM8</td>
      <td>ILE</td>
      <td>61.18</td>
      <td>58.27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1BM8</td>
      <td>TYR</td>
      <td>56.95</td>
      <td>56.18</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1BM8</td>
      <td>SER</td>
      <td>56.35</td>
      <td>56.84</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1BM8</td>
      <td>ALA</td>
      <td>51.96</td>
      <td>51.01</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1BM8</td>
      <td>ARG</td>
      <td>56.54</td>
      <td>54.64</td>
    </tr>
  </tbody>
</table>
</div>




```python
diff = cs_data.theo.values - cs_data.exp.values
idx = pd.Categorical(cs_data['aa']).codes
groups = len(np.unique(idx))
print(f'There are {groups} different amino acids in the data.')
```

    There are 19 different amino acids in the data.


- for comparison, we will build both a non-hierarchical model and a hierarchical model


```python
with pm.Model() as cs_nh:
    µ = pm.Normal('µ', mu=0, sd=10, shape=groups)
    σ = pm.HalfNormal('σ', sd=10, shape=groups)
    
    y = pm.Normal('y', mu=µ[idx], sd=σ[idx], observed=diff)
    
    trace_cs_nh = pm.sample(1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [σ, µ]




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
  100.00% [4000/4000 00:13<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 21 seconds.



```python
with pm.Model() as cs_h:
    # hyper-priors on varying µ
    µ_µ = pm.Normal('µ_µ', mu=0, sd=10)
    σ_µ = pm.HalfNormal('σ_µ', 10)
    
    # priors
    µ = pm.Normal('µ', mu=µ_µ, sd=σ_µ, shape=groups)
    σ = pm.HalfNormal('σ', sd=10, shape=groups)
    
    y = pm.Normal('y', mu=µ[idx], sd=σ[idx], observed=diff)
    
    trace_cs_h = pm.sample(1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [σ, µ, σ_µ, µ_µ]




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
  100.00% [4000/4000 00:19<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 29 seconds.


- compare the models using `plot_forest()`


```python
axes = az.plot_forest(
    [az.from_pymc3(trace_cs_nh), az.from_pymc3(trace_cs_h)],
    model_names=['n_h', 'h'],
    var_names='µ',
    combined=True,
    colors='cycle'
)

y_lims = axes[0].get_ylim()
axes[0].vlines(trace_cs_h['µ_µ'].mean(), *y_lims)

plt.show()
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/arviz/data/io_pymc3.py:85: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      warnings.warn(



![png](02_programming-probabilistically_files/02_programming-probabilistically_70_1.png)



```python

```
