---
title: "Using Schatten-p norms to dynamically interpolate between Adam and Muon"
date: 2026-02-19 12:00:00 +0100
tags: [optimization]
categories: [blog]
layout: posts
---

# Using Schatten-p norms to dynamically interpolate between Adam and Muon
Thomas MASSENA

<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
TLDR: Using an adaptive scheme allows a theoretically principled way to choose Adam-esque or Muon-esque weight updates.
</div>

### A Bit of Context

Since 2014-ish, the Adam optimizer has become the de-facto gradient-based optimization method for Deep Neural Networks (DNNs). Recently however, a concurrent optimizer named Muon has made headlines for its consistent gains over Adam in a variety of different scenarios. Muon uses the following update rule on 2D parameters after momentum accumulation:

$$
 W_{t+1} = W_t - \eta_t . \mathrm{PolarFactor}(M_t)
$$

with $M_t$ the update after momentum accumulation of Singular Value Decomposition (SVD) $U.\Sigma.V^T$. The $\mathrm{PolarFactor}$ function returns to the closest orthogonal matrix to $M_t$, i.e.: 

$$ 
 \mathrm{PolarFactor}(M_t) = U.V^T
$$

which an be efficiently approximated via a Newton-Schulz iterative scheme. Also, as described in Bernstein and Newhouse's "Old Optimizer, New Norm" (OONN) paper. Muon (and Shampoo), without E.M.A correspond to the scaled solution of the following steepest descent direction problem:

$$
 \mathrm{argmin}_{\Delta G \in \mathbb{R}^{m,n}} \langle G, \Delta W \rangle_F + \frac{\lambda}{2} . \| \Delta W \|_2^2.
$$

meaning that the Muon udpate is a spectral descent direction under the maximum spectral norm update constraint (controlled by $\lambda$).

### Interpolating Between SGD and Muon

Now, given an unpreconditioned update $M_t$ of SVD $U.\Sigma.V^T$, let's propose the following update rule:

$$
 W_{t+1} = W_t - \eta_t . ( U.\Sigma^{1/p}.V^T ), \forall \ p \in [1, \infty),
$$

which recovers the SGD update rule when $p=1$ and Muon when $p \rightarrow \infty$. 


Using the OONN framework, we find this update rule to be a steepest descent direction as it corresponds to the scaled version of solution to the following problem:

$$
 \mathrm{argmin}_{\Delta W \in \mathbb{R}^{m,n}} \langle G, \Delta W \rangle_F + \frac{\lambda}{p+1}. \| \Delta W \|_{p+1}^{p+1}
$$

where $\| \cdot \|_p$ denotes the Schatten-p norm (i.e. the $\ell_p$ norm of singular values).


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #c1c1e1;">
 Note: independently from my approach, <a href="https://www.arxiv.org/pdf/2602.04669" target="_blank">(Qi et al, 2026)</a> actually propose optimization methods that use statically set $p=\{1, 2, 4, \infty\}$ variants across training runs.
</div>

<br>

<div class="proof-block" style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #007BFF; margin-bottom: 20px;">

<p><strong>Proof Intuition:</strong></p>

<p>Because both the Frobenius inner product and the Schatten norm are unitarily invariant, we can completely decouple the matrix optimization problem into independent, 1D scalar minimizations for each singular value. Let the gradient's singular values be <em>&sigma;<sub>i</sub></em> and our update step's singular values be <em>s<sub>i</sub></em>. The objective function simplifies to:</p>

$$
\min_{s_i} \left( -\sigma_i s_i + \frac{\lambda}{p+1} s_i^{p+1} \right)
$$

<p>Taking the derivative with respect to <em>s<sub>i</sub></em> and setting it to zero gives:</p>

$$
-\sigma_i + \lambda s_i^p = 0 \implies s_i = \left(\frac{\sigma_i}{\lambda}\right)^{1/p}
$$

<p>Absorbing the constant <em>(1/&lambda;)<sup>1/p</sup></em> into the learning rate <em>&eta;<sub>t</sub></em>, we are left with the step's singular values <em>s<sub>i</sub> &prop; &sigma;<sub>i</sub><sup>1/p</sup></em>. Reconstructing the matrix with these new singular values yields the <em>U &Sigma;<sup>1/p</sup> V<sup>T</sup></em> update. The behavior is straightforward:</p>

</div>

### Efficient Approximation

Well that's very cool, we now have a way to vary the value of $p$. However, how do we impose that in practice without using costly SVD computations ? The answer is actually quite easy. Newton-Schulz, again, for the win. The only mechanism we would really need to tune being the coefficient computation method. 

<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #c1c1e1;">
Note: For more insights on Newton-Schulz coefficient computation, you can refer to <a href="https://leloykun.github.io/ponder/muon-opt-coeffs/" target="_blank">Franz Cesista's method</a> or even the <a href="https://arxiv.org/abs/2506.10935" target="_blank">Grishina (CANS)</a> or <a href="https://arxiv.org/abs/2505.16932" target="_blank">Amsel (Polar Express)</a> papers.
</div>

<br>

So, back to our endeavour, how do we compute these coeffs ? For the sake of you and I, let's keep it stupidly simple, we want to compute coefficients $a$, $b$ and $c$, such that:

$$
 \min_{a,b,c \in \mathbb{R}^3} \int_0^1 (a . x + b . x^3 + c . x^5 - x^{1/p}) . dx \quad \mathrm{s.t} \quad a+b+c = 1.
$$

For better approximation performance, we can perform $N$ Newton-Schulz approximation steps and approximate $x_{t+1} = x_{t}^{p^{-1/N}}$ at each step. Anyways, this can be solved really efficiently using Lagrange multipliers. *This really simple method could probably benefit from further improving, for the sake of conciseness however we'll keep it at that*. 

Now, we have a toolbox to compute Schatten-p norm update directions efficiently. Remains the most important question:

<figure style="text-align: center;">
  <img src="/assets/images/pMuon/y-tho.jpg"
       style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.4em; color: #666;">
    <strong> Why should we care? </strong>
  </figcaption>
</figure>

### Theoretically Principled Approach

For this, we rely on the exhaustive work of <a href="https://arxiv.org/abs/2512.04299" target="_blank">Davis and Drusvyatskiy, 2025</a>, which answers the question: *"when do spectral gradient updates help in deep learning?"*. 

In the setting of random feature regression on $n$ points, they define the objective function as:

$$
 \mathcal{L}(W) = \frac{1}{2n} . \| W. A - Y \|_F^2.
$$

where $W$ is the weight matrix, $A$ the post activation matrix and $Y$ the random feature to be regressed. A simple Taylor-expansion yields, for any update $\delta W \in \mathbb{R}^{m,n}$, we have:

$$
 \mathcal{L}(W + \delta W) = \mathcal{L} + \langle \nabla \mathcal{L}(W), \delta W \rangle + \frac{1}{2n} \| \delta W . A \|_F^2.
$$

Where, to upper bound $ \| \delta W . A \|_F^2 $, SGD-like gradient descent relies on: $\| \delta W . A \|_F^2 \leq \| \delta W \|_F . \| A \|_2 $.
And Muon-like (spectral) gradient descent relies on: $\| \delta W . A \|_F^2 \leq \| \delta W \|_2 . \| A \|_F $. In this framework, both updates recover:

$$
\begin{aligned}
\mathcal{L}(W) - \mathcal{L}(W_{\mathrm{SGD}})
&\geq \frac{1}{2 L_F}\,\|G\|_F^2 \\
\text{with:}\quad
L_F &= \frac{1}{n}\,\|A\|_2^2 \\[8pt]
%
\mathcal{L}(W) - \mathcal{L}(W_{\mathrm{Muon}})
&\geq \frac{1}{2 L_2}\,\|G\|_F^2 \\
\text{with:}\quad
L_2 &= \frac{1}{n}\,\|A\|_F^2
\end{aligned}
$$

Ideally, we want to find a formula $f(G, A) = p$, based on gradients and activations, to use the optimal $p$ value for our update. Thus garanteeing better performance ! To this end, we rely on the generalized Hölder inequality, which states, $\exists k \in [1, \infty)$, s.t:

$$
 \| \delta W . A \|_F \leq \| \delta G \|_{p+1} . \| A \|_k
$$

with condition, $\frac{1}{2} = \frac{1}{p+1} + \frac{1}{k}$, with the $\frac{1}{2}$ factor stemming from the original square on $\| \delta W . A \|_{F}^{2}$. 

Therefore, we recover, $L_p = \frac{1}{n} . \| A \|_{k(p)}^2$ as the Lipschitz constant for this update geometry. Using this, we find:

$$
\begin{aligned}
\mathcal{L}(W) - \mathcal{L}(W_{\mathrm{pMuon}})
&\geq \frac{1}{2 L_p}\,\|G\|_{1 + 1/p}^2 \\
\end{aligned}
$$

with $1 + 1/p$ the dual norm of $p+1$-th Schatten norm. Which gives the following optimality condition:

$$
 p^* = \mathrm{argmax}_{p \in [1, \infty)} \left( \frac{\| G \|_{1 + 1/p}}{ \| A \|_{\frac{2 (p+1)}{p-1}}} \right)
$$

Nice ! We now have sort of an "*optimality*" condition on the value of $p$ for one step ! **There is a problem however, this computation requires knowing the full singular value spectral of** $A$ **and** $G$.

### Computing this Efficiently

Right now, anyone who has already trained big neural networks and ran into runtime or memory issues is probably cringing, knowing that the computation of the full singular value spectra of both the raw gradient and the post-activations is computational suicide.

<figure style="text-align: center;">
  <img src="/assets/images/clenched-fist.png"
       style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.2em; color: #666;"> <em> Probably your initial reaction ? </em>
  </figcaption>
</figure>

Luckily, several elements can mitigate the computational overhead of using this method. Without going into too much detail:
(i) we can run the $p$ updates every N steps (ii) we can run the computation of our optimal $p$ using subsampled $G$ and $A$ matrices with an extra spectral norm computation. To counteract the potentially high variance of our singular value estimators, we use exponential moving average of the singular values across updates, then we compute the $p$ update.

Also, finally, our iterative scheme to update $p$ can be run on a separate process from the training run, which will update $p$ between training steps when a new $p$ value is recovered.

So to summarize, if we subsample the gradient and activation matrices by a factor $S$, the extra overhead required by our algorithm is:
- **Runtime**: None if the $p$ update does not hold up the training process.
- **Memory**: requires storing a $S \times \mathrm{m,n}$ vector containing randomly sampled singular values.
- **Bandwidth**: In a multi-gpu setting the subsampled singular value vectors should probably be synchronised.

In practice, if the optimal $p$ value is a relatively stable quantity, the fact that it's update lags behind the training process should not be too much of an issue. 

### Algorithm

Our algorithm takes up the following form:

<div style="border:1px solid #ccc; padding:16px; border-radius:8px; background:#f9f9f9;">

<strong>Adaptive Schatten-p Update (layerwise)</strong>

<ol>
  <li>Finish current training step upon receiving new <em>p</em>.</li>
  <li>Collect minibatch activations.</li>
  <li>Collect gradients.</li>
  <li>Subsample activation and gradient matrices.</li>
  <li>Update EMAs of subsampled singular values.</li>
  <li>Compute updated <em>p</em> from subsampled spectra.</li>
  <li>Dispatch spectra computation to a separate thread.</li>
</ol>

</div>

### Empirical Results

To check out the validity of our implementation, we run a gridsearch on a ResMLP neural network on the CIFAR-10 dataset. 

<figure style="text-align: center;">
  <img src="/assets/images/pMuon/resmpl_cifar.png"
       style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.4em; color: #666;">
    <strong> ResMLP train loss on cifar-10 after 50 epochs, using 3 different random seeds. </strong>
  </figcaption>
</figure>

So yay I guess, our version is competitive ! Now let's get to the fun stuff. 

Given that our optimizer is now adaptative, what insights do we recover about the optimal $p$ update across layers. For example, on that same ResMLP network, we notice:


### References


