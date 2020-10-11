# Ch 2. Generalized Linear Models


```python
import numpy as np
import pandas as pd
from scipy import stats
import pymc3 as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns
```

## Generalized linear models

## Logistic regression

- GLM with a *logistic function* for the inverse link function
    - the output for any input $z$ will lie within the 0 and 1 interval
    - thus, transforms data for a Bernoulli distribution

$$
\text{logistic}(z) = \frac{1}{1 + e^{-z}}
$$


```python
z = np.linspace(-8, 8)
y = 1 / (1 + np.exp(-z))
plt.plot(z, y)
plt.xlabel("z")
plt.ylabel("y = logisitc(z)")
plt.show()
```


![png](04_generalized-linear-models_files/04_generalized-linear-models_3_0.png)


### The logistic model

- same as a linear regression, just using the logistic link function and the Bernoulli likelihood function

$$
\theta = logistic(\alpha + x \beta) \\
y \sim \text{Bernoulli}(\theta)
$$

### The Iris dataset


```python
iris = pd.read_csv("data/iris.csv")
iris.head()
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
      <th>sepal_length</th>
      <th>sepal_width</th>
      <th>petal_length</th>
      <th>petal_width</th>
      <th>species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
sns.stripplot(x="species", y="sepal_length", data=iris, jitter=True)
plt.show()
```


![png](04_generalized-linear-models_files/04_generalized-linear-models_7_0.png)



```python
sns.pairplot(iris, hue="species", diag_kind="kde")
plt.show()
```


![png](04_generalized-linear-models_files/04_generalized-linear-models_8_0.png)


#### The logistic model applied to the Iris dataset

- classify `setosa` vs. `versicolor` using `sepal_length` as the only predictor
    - encode the species as 0 and 1, respectively


```python
df = iris.query("species == ('setosa', 'versicolor')")
y_0 = pd.Categorical(df["species"]).codes
x_n = "sepal_length"
x_0 = df[x_n].values
x_c = x_0 - x_0.mean()  # center the data
```

- two deterministic variables in this model:
    - `θ`: ouput of the logistic function applied to `µ`
    - `bd`: the "boundary decision" is the value of the predictor variable used to separate classes


```python
with pm.Model() as model_0:
    α = pm.Normal("α", mu=0, sd=10)
    β = pm.Normal("β", mu=0, sd=10)

    µ = α + pm.math.dot(x_c, β)
    θ = pm.Deterministic("θ", pm.math.sigmoid(µ))
    bd = pm.Deterministic("bd", -α / β)

    y1 = pm.Bernoulli("y1", p=θ, observed=y_0)

    trace_0 = pm.sample(1000)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [β, α]




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



    Sampling 2 chains for 1_000 tune and 1_000 draw iterations (2_000 + 2_000 draws total) took 14 seconds.



```python
az_trace_0 = az.from_pymc3(trace_0, model=model_0)
az.plot_trace(az_trace_0, var_names=["α", "β"])
plt.show()
```


![png](04_generalized-linear-models_files/04_generalized-linear-models_13_0.png)



```python
az.summary(az_trace_0)
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
      <th>α</th>
      <td>0.295</td>
      <td>0.332</td>
      <td>-0.347</td>
      <td>0.873</td>
      <td>0.009</td>
      <td>0.007</td>
      <td>1304.0</td>
      <td>1249.0</td>
      <td>1297.0</td>
      <td>1127.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>β</th>
      <td>5.339</td>
      <td>1.001</td>
      <td>3.342</td>
      <td>7.085</td>
      <td>0.026</td>
      <td>0.019</td>
      <td>1480.0</td>
      <td>1448.0</td>
      <td>1509.0</td>
      <td>1487.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>θ[0]</th>
      <td>0.164</td>
      <td>0.053</td>
      <td>0.071</td>
      <td>0.264</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1381.0</td>
      <td>1381.0</td>
      <td>1360.0</td>
      <td>1311.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>θ[1]</th>
      <td>0.068</td>
      <td>0.034</td>
      <td>0.016</td>
      <td>0.132</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1473.0</td>
      <td>1473.0</td>
      <td>1415.0</td>
      <td>1328.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>θ[2]</th>
      <td>0.027</td>
      <td>0.019</td>
      <td>0.002</td>
      <td>0.060</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>1520.0</td>
      <td>1520.0</td>
      <td>1455.0</td>
      <td>1355.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>θ[96]</th>
      <td>0.811</td>
      <td>0.067</td>
      <td>0.685</td>
      <td>0.926</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1329.0</td>
      <td>1329.0</td>
      <td>1377.0</td>
      <td>1206.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>θ[97]</th>
      <td>0.979</td>
      <td>0.018</td>
      <td>0.948</td>
      <td>0.999</td>
      <td>0.001</td>
      <td>0.000</td>
      <td>1220.0</td>
      <td>1220.0</td>
      <td>1468.0</td>
      <td>1251.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>θ[98]</th>
      <td>0.164</td>
      <td>0.053</td>
      <td>0.071</td>
      <td>0.264</td>
      <td>0.001</td>
      <td>0.001</td>
      <td>1381.0</td>
      <td>1381.0</td>
      <td>1360.0</td>
      <td>1311.0</td>
      <td>1.01</td>
    </tr>
    <tr>
      <th>θ[99]</th>
      <td>0.811</td>
      <td>0.067</td>
      <td>0.685</td>
      <td>0.926</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1329.0</td>
      <td>1329.0</td>
      <td>1377.0</td>
      <td>1206.0</td>
      <td>1.00</td>
    </tr>
    <tr>
      <th>bd</th>
      <td>-0.053</td>
      <td>0.061</td>
      <td>-0.164</td>
      <td>0.066</td>
      <td>0.002</td>
      <td>0.001</td>
      <td>1252.0</td>
      <td>1247.0</td>
      <td>1249.0</td>
      <td>1135.0</td>
      <td>1.00</td>
    </tr>
  </tbody>
</table>
<p>103 rows × 11 columns</p>
</div>




```python
theta = trace_0["θ"].mean(axis=0)
idx = np.argsort(x_c)
plt.plot(x_c[idx], theta[idx], color="C2", lw=3)
plt.vlines(trace_0["bd"].mean(), 0, 1, color="k")
bd_hpi = az.hdi(trace_0["bd"])
plt.fill_between(bd_hpi, 0, 1, color="k", alpha=0.25)
plt.scatter(x_c, np.random.normal(y_0, 0.02), marker=".", color=[f"C{x}" for x in y_0])
az.plot_hdi(x_c, trace_0["θ"], color="C2", ax=plt.gca())

plt.xlabel(x_n)
plt.ylabel("θ", rotation=0)
locs, _ = plt.xticks()
plt.xticks(locs, np.round(locs + x_0.mean(), 1))

plt.show()
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/arviz/stats/stats.py:483: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
      warnings.warn(



![png](04_generalized-linear-models_files/04_generalized-linear-models_15_1.png)


## Multiple logistic regression


```python
df = iris.query("species == ('setosa', 'versicolor')")
y_1 = pd.Categorical(df.species).codes
x_n = ["sepal_length", "sepal_width"]
x_1 = df[x_n].values
```

### The boundary decision

$$
0.5 = \text{logistic}(\alpha + \beta_1 x_1 + \beta_2 x_2) \Leftrightarrow 0 = \alpha + \beta_1 x_1 + \beta_2 x_2 \\
x_2 = - \frac{\alpha}{\beta_2} - \frac{\beta_1}{\beta_2}x_1
$$

### Implementing the model


```python
with pm.Model() as model_1:
    alpha = pm.Normal("alpha", mu=0, sd=10)
    beta = pm.Normal("beta", mu=0, sd=2, shape=len(x_n))

    mu = alpha + pm.math.dot(x_1, beta)
    theta = pm.Deterministic("theta", 1 / (1 + pm.math.exp(-mu)))
    bd = pm.Deterministic(
        "bd", -1.0 * alpha / beta[1] - (x_1[:, 0] * beta[0] / beta[1])
    )

    y1 = pm.Bernoulli("y1", p=theta, observed=y_1)

    trace_1 = pm.sample(2000)

az_trace_1 = az.from_pymc3(trace=trace_1, model=model_1)
```

    Auto-assigning NUTS sampler...
    Initializing NUTS using jitter+adapt_diag...
    Multiprocess sampling (2 chains in 2 jobs)
    NUTS: [beta, alpha]




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
  100.00% [6000/6000 00:29<00:00 Sampling 2 chains, 0 divergences]
</div>



    Sampling 2 chains for 1_000 tune and 2_000 draw iterations (2_000 + 4_000 draws total) took 36 seconds.



```python
az.plot_trace(az_trace_1, var_names=["alpha", "beta"])
plt.show()
```


![png](04_generalized-linear-models_files/04_generalized-linear-models_21_0.png)



```python
idx = np.argsort(x_1[:, 0])
bd = trace_1["bd"].mean(0)[idx]
plt.scatter(x_1[:, 0], x_1[:, 1], c=[f"C{x}" for x in y_0])
plt.plot(x_1[:, 0][idx], bd, color="k")
az.plot_hdi(x_1[:, 0], trace_1["bd"], color="k", ax=plt.gca())
plt.xlabel(x_n[0])
plt.ylabel(x_n[1])
plt.show()
```

    /Users/admin/Developer/Python/bayesian-analysis-with-python_e2/.env/lib/python3.8/site-packages/arviz/stats/stats.py:483: FutureWarning: hdi currently interprets 2d data as (draw, shape) but this will change in a future release to (chain, draw) for coherence with other functions
      warnings.warn(



![png](04_generalized-linear-models_files/04_generalized-linear-models_22_1.png)


### Interpreting the coefficients of a logisitic regression

- because the model uses a non-linear link function, the effect of $\beta$ is a non-linear function on $x$
    - if $\beta$ is positive, then as $x$ increases, so to does $\Pr(y=1)$, but non-linearly
- some algebra to understand the effect of a coefficient

$$
\theta = \text{logistic}(\alpha + X \beta) \quad \text{logit}(z) = \log(\frac{z}{1-z}) \\
\text{logit}(\theta) = \alpha + X \beta \\
\log(\frac{\theta}{1-\theta}) = \alpha + X \beta \\
\log(\frac{\Pr(y=1)}{1-\Pr(y=1)}) = \alpha + X \beta
$$

- recall: $\frac{\Pr(y=1)}{1 - \Pr(y=1)}$ = **"odds"**
    - *"In a logistic regression, the $\beta$ coefficient encodes the increase in log-odds units by unit increase of the $x$ variable."*

### Dealing with correlated variables




```python

```
