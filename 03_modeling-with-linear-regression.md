# Ch 3. Modeling with Linear Regression

- topics covered:
    - simple linear regression
    - robust linear regression
    - hierarchical linear regression
    - polynomial regression
    - multiple linear regression
    - interactions
    - variable variance


```python
import numpy as np
import pandas as pd
from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
```

## Simple linear regression

### The machine learning connection

### The core of the linear regression models

- can treat solving of linear regression multiple ways:
    - OLS has the goal of minimizing a loss function
    - Bayesian uses a probabilizistic approach

$$y \sim \mathcal{N}(\mu = \alpha + x \beta, \epsilon)$$

>  A linear regression model is an extension of the Gaussian model where the mean is not directly estimated but rather computed as a linear function of a predictor variable and some additional parameters.

- set priors for the variables in the equation

$$
\alpha \sim \mathcal{N}(\mu_\alpha, \sigma_\alpha) \\
\beta \sim \mathcal{N}(\mu_\beta, \sigma_\beta) \\
\epsilon \sim | \mathcal{N}(0, \sigma_\epsilon) |
$$

<img src="assets/ch03/c5c481c3-353f-4b8f-9764-d166813e263e.png" width="75%">


- an example of fitting a linear regression with fake data


```python
np.random.seed(1)
N = 100
alpha_real = 2.5
beta_real = 0.9
eps_real = np.random.normal(0, 0.5, size=N)

x = np.random.normal(10, 1, N)
y_real = alpha_real + beta_real*x
y = y_real + eps_real

fig, ax = plt.subplots(1, 2, figsize=(8, 4))

ax[0].plot(x, y, 'C0.')
ax[0].set_xlabel('x')
ax[0].set_ylabel('y', rotation = 0)
ax[0].plot(x, y_real, 'k')

az.plot_kde(y, ax=ax[1])
ax[1].set_xlabel('y')

plt.tight_layout()
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_6_0.png)



```python
with pm.Model() as model_g:
    α = pm.Normal('α', mu=0, sd=10)
    β = pm.Normal('β', mu=0, sd=1)
    ϵ = pm.HalfCauchy('ϵ', 5)
    
    µ = pm.Deterministic('µ', α + β*x)
    y_pred = pm.Normal('y_pred', mu=µ, sd=ϵ, observed=y)
    
    trace_g = pm.sample(2000, tune=1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [ϵ, β, α]




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
  100.00% [6000/6000 00:29<00:00 Sampling 2 chains, 65 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 38 seconds.
    The acceptance probability does not match the target. It is 0.885111829718544, but should be close to 0.8. Try to increase the number of tuning steps.
    There were 65 divergences after tuning. Increase `target_accept` or reparameterize.
    The acceptance probability does not match the target. It is 0.6965599556729883, but should be close to 0.8. Try to increase the number of tuning steps.
    The estimated number of effective samples is smaller than 200 for some parameters.



```python
az_trace_g = az.from_pymc3(trace_g, model=model_g)
az.plot_trace(az_trace_g, var_names=['α', 'β', 'ϵ'])
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_8_0.png)



```python
az.plot_autocorr(az_trace_g, var_names=['α', 'β', 'ϵ'])
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_9_0.png)



```python
az.plot_forest(az_trace_g, var_names=['α', 'β', 'ϵ'])
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_10_0.png)



```python
with model_g:
    model_g_ppc = pm.sample_posterior_predictive(trace_g)
```



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
  100.00% [4000/4000 00:06<00:00]
</div>




```python
az.plot_ppc(az.from_pymc3(trace_g, posterior_predictive=model_g_ppc),
            num_pp_samples=100)
plt.show()
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/arviz/data/io_pymc3.py:85: FutureWarning: Using `from_pymc3` without the model will be deprecated in a future release. Not using the model will return less accurate and less useful results. Make sure you use the model argument or call from_pymc3 within a model context.
      warnings.warn(



![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_12_1.png)


### Linear models and high autocorrelation

- the posterior distribution of $\alpha$ and $\beta$ are highly correlated as a matter of definition
    - this results in a diagonal posterior space that can be problematic for the sampling process
    - this will be discussed further in later chapters


```python
az.plot_pair(az_trace_g, var_names=['α', 'β'], scatter_kwargs={'alpha': 0.1})
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_14_0.png)


#### Modifying the data before running

- centering and scaling the data can help turn the diagonal posterior into a more circular form
    - this is usually better for the sampling process

### Interpreting and visualizing the posterior


```python
# Plot lines sample from the posterior.
draws = range(0, len(trace_g['α']), 10)
plt.plot(x, trace_g['α'][draws] + trace_g['β'][draws] * x[:, np.newaxis],
         c = 'gray', alpha = 0.3)

# Plot line from average of posterior.
alpha_m = trace_g['α'].mean()
beta_m = trace_g['β'].mean()
plt.plot(x, alpha_m + beta_m*x, c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')

# Plot original data.
plt.plot(x, y, 'C0.')

plt.xlabel('x', fontsize=16)
plt.ylabel('y', rotation=0, fontsize=16)
plt.legend(fontsize=12)
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_17_0.png)


- plot the **highest density interval (HDI)**


```python
plt.plot(x, alpha_m + beta_m * x, c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
for ci, c in zip([0.95, 0.89, 0.75, 0.5], ["#cfe9ff", "#99d1ff", "#52b1ff", "#058fff"]):
    az.plot_hdi(x=x, hdi_data=az.hdi(az_trace_g, hdi_prob=ci)['µ'],
                color=c, ax=plt.gca())
plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.legend()
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_19_0.png)


- also plot the HDI for $\hat{y}$
    - where the model expects to see the given percent of the data


```python
ppc = pm.sample_posterior_predictive(trace_g, samples=2000, model=model_g)
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
  <progress value='2000' class='' max='2000' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [2000/2000 00:02<00:00]
</div>




```python
plt.plot(x, y, 'b.')
plt.plot(x, alpha_m + beta_m*x, c='k',
         label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
az.plot_hdi(x=x, hdi_data=az.hdi(ppc['y_pred'], hdi_prob=0.5), color='black', ax=plt.gca())
az.plot_hdi(x=x, hdi_data=az.hdi(ppc['y_pred'], hdi_prob=0.95), color='gray', ax=plt.gca())

plt.xlabel('x', fontsize=16)
plt.ylabel('y', fontsize=16)
plt.show()
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/arviz/stats/stats.py:483: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
      warnings.warn(
    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/arviz/stats/stats.py:483: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
      warnings.warn(



![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_22_1.png)


### Pearson correlation coefficient

#### Pearson coefficient from a multivariate Gaussian

### Robust linear regression

### Hierarchical linear regression

- create 8 related data groups
    - one group will only have 1 data point


```python
N = 20
M = 8
idx = np.repeat(range(M-1), N)
idx = np.append(idx, 7)
np.random.seed(314)

alpha_real = np.random.normal(2.5, 0.5, size=M)
beta_real = np.random.beta(6, 1, size=M)
eps_real = np.random.normal(0, 0.5, size=len(idx))

y_m = np.zeros(len(idx))
x_m = np.random.normal(10, 1, len(idx))
y_m = alpha_real[idx] + beta_real[idx] * x_m + eps_real

_, ax = plt.subplots(2, 4, figsize=(10, 5), sharex=True, sharey=True)
ax = np.ravel(ax)

j = 0
k = N
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', rotation=0, labelpad=15)
    ax[i].set_xlim(6, 15)
    ax[i].set_ylim(7, 17)
    j += N
    k += N

plt.tight_layout()
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_24_0.png)


- center the data before fitting model


```python
x_centered = x_m - x_m.mean()
```

- fit a non-hierarchical model for comparison
    - there is also a line that rescales $\alpha$ to adjust for the centering


```python
with pm.Model() as unpooled_model:
    α_temp = pm.Normal('α_temp', mu=0, sd=10, shape=M)
    β = pm.Normal('β', mu=0, sd=10, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)
    
    y_pred = pm.StudentT('y_pred', mu=α_temp[idx] + β[idx]*x_centered,
                         sd=ϵ, nu=ν, observed=y_m)
    
    α = pm.Deterministic('α', α_temp - β*x_m.mean())
    
    trace_up = pm.sample(2000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [ν, ϵ, β, α_temp]




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
  100.00% [6000/6000 00:20<00:00 Sampling 2 chains, 4 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 29 seconds.
    There was 1 divergence after tuning. Increase `target_accept` or reparameterize.
    There were 3 divergences after tuning. Increase `target_accept` or reparameterize.



```python
az_trace_up = az.from_pymc3(trace_up, model=unpooled_model)
az.plot_forest(az_trace_up, var_names=['α', 'β'], combined=True)
```




    array([<AxesSubplot:title={'center':'94.0% HDI'}>], dtype=object)




![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_29_1.png)


- now fit a hierarchical model with priors on the parameters of $\alpha$ and $\beta$

<img src="assets/ch03/hierarchcical-model-diagram.png" width="75%">


```python
with pm.Model() as hierarchical_model:
    # Hyper-priors
    α_µ_temp = pm.Normal('α_µ_temp', mu=0, sd=10)
    α_σ_temp = pm.HalfNormal('α_σ_temp', sd=10)
    β_µ = pm.Normal('β_µ', mu=0, sd=10)
    β_σ = pm.HalfNormal('β_σ', sd=10)
    
    # Priors
    α_temp = pm.Normal('α_temp', mu=α_µ_temp, sd=α_σ_temp, shape=M)
    β = pm.Normal('β', mu=β_µ, sd=β_σ, shape=M)
    ϵ = pm.HalfCauchy('ϵ', 5)
    ν = pm.Exponential('ν', 1/30)
    
    # Likelihood
    y_pred = pm.StudentT('y_pred',
                        mu=α_temp[idx] + β[idx] * x_centered,
                        sd=ϵ, nu=ν, observed=y_m)
    
    α = pm.Deterministic('α', α_temp - β*x_m.mean())
    α_µ = pm.Deterministic('α_µ', α_µ_temp - β_µ * x_m.mean())
    α_σ = pm.Deterministic('α_sd', α_σ_temp - β_µ * x_m.mean())
    
    trace_hm = pm.sample(1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [ν, ϵ, β, α_temp, β_σ, β_µ, α_σ_temp, α_µ_temp]




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
  100.00% [4000/4000 00:21<00:00 Sampling 2 chains, 60 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 32 seconds.
    There were 37 divergences after tuning. Increase `target_accept` or reparameterize.
    There were 23 divergences after tuning. Increase `target_accept` or reparameterize.
    The number of effective samples is smaller than 25% for some parameters.



```python
az_trace_hm = az.from_pymc3(trace_hm, model = hierarchical_model)
az.plot_forest(az_trace_hm, var_names=['α', 'β'], combined=True)
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_32_0.png)



```python
fix, ax = plt.subplots(2, 4, figsize=(10, 5), 
                       sharex=True, sharey=True, 
                       constrained_layout=True)
ax = np.ravel(ax)

j, k = 0, N
x_range = np.linspace(x_m.min(), x_m.max(), 10)
for i in range(M):
    ax[i].scatter(x_m[j:k], y_m[j:k])
    ax[i].set_xlabel(f'x_{i}')
    ax[i].set_ylabel(f'y_{i}', labelpad=17, rotation = 0)
    alpha_m = trace_hm['α'][:, i].mean()
    beta_m = trace_hm['β'][:, i].mean()
    ax[i].plot(x_range, alpha_m + beta_m * x_range,
               c='k',
               label=f'y = {alpha_m:.2f} + {beta_m:.2f} * x')
    ax[i].legend()
    plt.xlim(x_m.min()-1, x_m.max()+1)
    plt.ylim(y_m.min()-1, y_m.max()+1)
    j += N
    k += N

plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_33_0.png)


### Correlation, causation, and the messiness of life

## Polynomial regression

- example of fitting a polynomial regression


```python
ans = pd.read_csv('data/anscombe.csv')
x_2 = ans[ans.group == "II"]['x'].values
y_2 = ans[ans.group == "II"]['y'].values
x_2 = x_2 - x_2.mean()

plt.scatter(x_2, y_2)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
```




    Text(0, 0.5, 'y')




![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_35_1.png)



```python
with pm.Model() as model_poly:
    α = pm.Normal('α', mu=y_2.mean(), sd=1)
    β1 = pm.Normal('β1', mu=0, sd=1)
    β2 = pm.Normal('β2', mu=0, sd=1)
    ϵ = pm.HalfCauchy('ϵ', 5)
    
    µ = pm.Deterministic('µ', α + β1*x_2 + β2*(x_2**2))
    
    y_pred = pm.Normal('y_pred', mu=µ, sd=ϵ, observed=y_2)
    
    trace_poly = pm.sample(2000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [ϵ, β2, β1, α]




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
  100.00% [6000/6000 00:18<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 27 seconds.



```python
x_p = np.linspace(-6, 6)
y_p = trace_poly['α'].mean() + trace_poly['β1'].mean() * \
    x_p + trace_poly['β2'].mean() * x_p**2
plt.scatter(x_2, y_2)
plt.xlabel('x')
plt.ylabel('y', rotation=0)
plt.plot(x_p, y_p, c='C1')
plt.show()
```


![png](03_modeling-with-linear-regression_files/03_modeling-with-linear-regression_37_0.png)


### Interpreting the parameters of a polynomial regression

### Polynomial regression – the ultimate model?

## Multiple linear regression


```python

```
