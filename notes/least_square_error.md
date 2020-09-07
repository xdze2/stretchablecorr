from https://stats.stackexchange.com/a/428570

## Background

You can look through the slides here, but I will explain it as best as I can. You want the standard errors of the best-fit parameters, which is the same as the standard deviation of the best-fit parameters. The sd of the best fit parameters are given by the diagonal elements of the covariance matrix $\Sigma$. $\Sigma$ for non-linear regression is given by:

$$ \Sigma = \sigma^2 (H^{-1}) $$

where $\sigma$ is the standard deviation of the residuals and $H$ is the Hessian of the objective function (such as least squares or weighted least squares).
Finding standard deviation of the residuals, $\sigma$

If you don't know $\sigma$ from previous experiments, then you can estimate it as $\hat{\sigma}$ and use that estimated value to get $\Sigma = \hat{\sigma}^2 (H^{-1})$. It can be estimated with:

$$ \hat{\sigma} = \sqrt{\frac{f(x_{best})}{m-n}} $$

where $f(x_{best})$ is the best likelihood found by maximum-likelihood (aka best fit objective function). This can be something like the sum of squared residuals (SSE). $m$ is the number of parameters in your model. $n$ is the number of data points used to fit your model.




### Distribution of largest sample from normal distribution.

https://math.stackexchange.com/a/275419

The random variable $Z = \max_{i=1}^n(X_i)$ is known as order statistics, and is sometimes denoted as $X_{n:n}$.

The cumulative density function of $Z$ is easy to find:

$$
F_Z(z) = \mathbb{P}\left(Z \leqslant z\right) = \mathbb{P}\left(\max_{i=1}^n(X_i) \leqslant z\right) = \mathbb{P}\left( X_1 \leqslant z, X_2 \leqslant z, \ldots, X_n \leqslant z\right)
$$

using independence:

$$
F_Z(z) = \left(F_X(z)\right)^n
$$

Thus the density function is

$$
f_Z(z) = n f_X(z) F_X^{n-1}(z)
$$

In particular, it follows that $Z$ is not normal.

Expected values of $Z$ are known in closed form for $n=1,2,3,4,5$ (asking Mathematica):



and from https://math.stackexchange.com/a/89037

The $\max$-central limit theorem (Fisher-Tippet-Gnedenko theorem) can be used to provide a decent approximation when $n$ is large. See this example at reference page for extreme value distribution in Mathematica.

The $\max$-central limit theorem states that

$$
F_max(x) = \left(\Phi(x)\right)^n \approx F_{\text{EV}}\left(\frac{x-\mu_n}{\sigma_n}\right)
$$

, where $F_{EV} = \exp(-\exp(-x))$ is the cumulative distribution function for the extreme value distribution, and

$$
\mu_n = \Phi^{-1}\left(1-\frac{1}{n} \right) \qquad \qquad \sigma_n = \Phi^{-1}\left(1-\frac{1}{n} \cdot \mathrm{e}^{-1}\right)- \Phi^{-1}\left(1-\frac{1}{n} \right)
$$

Here $\Phi^{-1}(q)$ denotes the inverse cdf of the standard normal distribution.

The mean of the maximum of the size $n$ normal sample, for large $n$, is well approximated by

$$
m_n = \sqrt{2} \left((\gamma -1) \Phi^{-1}\left(2-\frac{2}{n}\right)-\gamma \Phi^{-1}\left(2-\frac{2}{e n}\right)\right)
$$

$$
= \sqrt{\log \left(\frac{n^2}{2 \pi \log \left(\frac{n^2}{2\pi} \right)}\right)} \cdot \left(1 + \frac{\gamma}{\log (n)} + \mathcal{o} \left(\frac{1}{\log (n)} \right) \right)
$$

where $\gamma$ is the Euler-Mascheroni constant.
