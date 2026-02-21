---
title: "Generalizing Muon to Schatten-p norms"
date: 2026-02-19 12:00:00 +0100
tags: [optimization]
categories: [blog]
layout: posts
---

# Using Schatten-p norms to dynamically interpolate between Adam and Muon
Thomas MASSENA
[Link to repository](https://github.com)


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
<strong>TLDR</strong>: Using an adaptive scheme Newton-Schulz allows a theoretically principled way to choose between SGD-<em>esque</em> or Muon-<em>esque</em> weight updates.
</div>

### A Bit of Context

Since 2014-ish, the Adam optimizer has become the de-facto gradient-based optimization method for Deep Neural Networks (DNNs). Recently however, a concurrent optimizer named Muon has made headlines for its consistent gains over Adam in a variety of different scenarios. 

Muon uses the following update rule on 2D parameters after momentum accumulation:

$$
 W_{t+1} = W_t - \eta_t . \mathrm{PolarFactor}(M_t)
$$

with $M_t \in \mathbb{R}^{m,n}$ the update after momentum accumulation, of Singular Value Decomposition (SVD): $U.\Sigma.V^T$, with $U$ and $V$ orthogonal (or semi-orthogonal) matrices. The $\mathrm{PolarFactor}$ function returns the closest orthogonal matrix to $M_t$, i.e.: 

$$ 
 \mathrm{PolarFactor}(M_t) = U.V^T
$$

Importantly, the computation of the Polar Factor can be efficiently approximated via a Newton-Schulz iterative scheme. And, as described in Bernstein and Newhouse's "Old Optimizer, New Norm" paper. Muon (and Shampoo), without E.M.A correspond to scaled solutions of the following steepest descent direction problem:

$$
 \mathrm{argmin}_{\Delta G \in \mathbb{R}^{m,n}} \langle G, \Delta W \rangle_F + \frac{\lambda}{2} . \| \Delta W \|_2^2.
$$

meaning that the Muon udpate is a spectral descent direction under the maximum spectral norm update constraint (controlled by sharpness parameter $\lambda$).

### Interpolating Between SGD and Muon

Now, given an unpreconditioned update $M_t$ of SVD $U.\Sigma.V^T$, let's propose the following update rule:

$$
 W_{t+1} = W_t - \eta_t . \left[ U.\Sigma^{1/p}.V^T \right], \forall \ p \in [1, \infty),
$$

which recovers the SGD update rule when $p=1$ and approaches Muon at exponential speed when $p \rightarrow \infty$. 


Using the "Old Optimizer, New Norm" framing, we find this update rule to be a steepest descent direction as it corresponds to the scaled version of solution to the following problem:

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

<p>Absorbing the constant <em>(1/&lambda;)<sup>1/p</sup></em> into the learning rate <em>&eta;<sub>t</sub></em>, we are left with the step's singular values <em>s<sub>i</sub> &prop; &sigma;<sub>i</sub><sup>1/p</sup></em>. Reconstructing the matrix with these new singular values yields the <em>U &Sigma;<sup>1/p</sup> V<sup>T</sup></em> update.</p>

</div>

### Efficient Approximation

Well, we now have a more general formulation of steepest descent directions that recovers both Muon and SGD. The main question now is: *can I compute $U.\Sigma^{1/p}.V^T$ efficiently ? How do we impose that in practice without using costly SVD computations ? The answer is actually quite easy. Newton-Schulz, again, for the win. Indeed, simply changing Newton-Schulz's coefficients suffices to approximate $U.\Sigma^{1/p}.V^T$ efficiently. 

<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #c1c1e1;">
Note: For more insights on Newton-Schulz coefficient computation, you can refer to <a href="https://leloykun.github.io/ponder/muon-opt-coeffs/" target="_blank">Franz Cesista's method</a> or even the <a href="https://arxiv.org/abs/2506.10935" target="_blank">Grishina (CANS)</a> or <a href="https://arxiv.org/abs/2505.16932" target="_blank">Amsel (Polar Express)</a> papers.
</div>

<br>

So, back to our endeavour, how do we compute these coeffs ? For the sake of you and I, let's keep it stupidly simple, we want to compute coefficients $a$, $b$ and $c$, such that:

$$
 \min_{a,b,c \in \mathbb{R}^3} \int_0^1 (a . x + b . x^3 + c . x^5 - x^{1/p}) . dx \quad \mathrm{s.t} \quad a+b+c = 1.
$$

For better approximation performance, we can perform $N$ Newton-Schulz approximation steps and approximate $x_{t+1} = x_{t}^{p^{-1/N}}$ at each step. Anyways, this can be solved really efficiently using Lagrange multipliers and solving this takes up negligeable time wrt a gradient step for a decently sized model. 

*This really simple method could probably benefit from further improving, for the sake of conciseness however we'll keep it at that*. 

Now, we have a toolbox to compute Schatten-p norm update directions efficiently. Remains the key question:

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

where $W$ is the weight matrix, $A$ the post activation matrix and $Y$ the random feature to be regressed. A simple Taylor-expansion yields, for any update $\delta W \in \mathbb{R}^{m,n}$, that:

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

Ideally, we want to find a formula $f(G, A) = p$, based on gradients and activations, that yields the optimal $p$ value to use for this step's update. Therefore offering a theoretical objective for better performance ! To this end, we rely on the generalized Hölder inequality, which states, $\exists k \in [1, \infty)$, s.t:

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

with $1 + 1/p$ the dual norm of $(p+1)$-th Schatten norm. Which gives the following optimality condition:

$$
 p^* = \mathrm{argmax}_{p \in [1, \infty)} \left( \frac{\| G \|_{1 + 1/p}}{ \| A \|_{\frac{2 (p+1)}{p-1}}} \right)
$$

Nice ! We now have sort of a layerwise "*optimality*" condition on the value of $p$ for one step ! **There is a problem however, this computation requires knowing the full singular value spectral of** $A$ **and** $G$.

### Computing this Efficiently

Right now, anyone who has already trained big neural networks and ran into runtime or memory issues is probably cringing, knowing that the computation of the full singular value spectra of both the raw gradient and the post-activations is computational suicide.

<figure style="text-align: center;">
  <img src="/assets/images/pMuon/clenched-fist.png"
       style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.2em; color: #666;"> <em> Probably your initial reaction ? </em>
  </figcaption>
</figure>

Luckily, several elements can mitigate the computational overhead of using this method. Without going into too much detail:
(i) we can run the $p$ updates every N steps (ii) we can run the computation of our optimal $p$ using subsampled $G$ and $A$ matrices with an extra spectral norm computation. (iii) we can group layers into sequential chunks where we will compute the mean optimal $p$ value to update with. To counteract the potentially high variance of our singular value estimators, we use exponential moving average of the singular values across updates, then we compute the $p$ update.

Also, finally, our iterative scheme to update $p$ can be run on a separate process from the training run, which will update $p$ between training steps when a new $p$ value is recovered.

So to summarize, if we subsample the gradient and activation matrices of $L$ sequential chunks by a factor $S$, the extra overhead required by our algorithm is:
- **Runtime**: None if the $p$ update does not hold up the training process.
- **Memory**: requires storing a $L \times S \times \mathrm{min}(m,n)$ vector containing randomly sampled singular values.
- **Bandwidth**: In a multi-gpu setting the subsampled singular value vectors should probably be synchronised across devices.

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
  <img src="/assets/images/pMuon/resmlp_cifar.png"
       style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.0em; color: #666;">
    <strong> ResMLP train loss on cifar-10 after 50 epochs, using 3 different random seeds. </strong>
  </figcaption>
</figure>

So yay I guess, our version is competitive ! For reference, tracking the optimal $p$ for all layers every 100 steps yields a 20s slowdown on a run that took the base Muon implementation 4 minutes and 25 seconds. This is pretty good given that this initial test was more about validating the concept than pushing for efficiency. Now let's get to the fun stuff. 

Given that our optimizer is now adaptative, what insights do we recover about the optimal $p$ update across layers ? For example, on that same ResMLP network, we notice:

<figure style="text-align: center;">
  <img src="/assets/images/pMuon/optimal_p_values_resmlp.png"
       style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.0em; color: #666;">
    <strong> The optimal p-value across layers can vary quite strongly along the depth of the ResMLP network, while staying stable. </strong>
  </figcaption>
</figure>

This recovers a typical empirical insight about the Muon optimizer, e.g., why people usually tend to exclude first and last layers from the Muon-optimized groups of parameters.

**More or less related parallel:** Although this is maybe unrelated, the relationship between neural network depth and optimal gradient rank can be somewhat related to the complexity-theory driven work of <a href="https://www.arxiv.org/pdf/2407.06076" target="_blank">Fel et al., 2024<a/>, where the authors show that higher complexity features are learned in the later layers of a ResNet50 model. This seems like this could be two sides of the same coin, or at least remotely related ?


### Limitations and Opportunities

While this whole study show that Muon orthogonalization can be made adaptative. I believe the following elements could still be improved:
- The coefficient computation method could probably be improved via Chebyshev-type polynomial algorithms (see the Amsel et al. & Grishina et al. papers).
- Deriving a the proper $\eta_{t,p}$ empirically or theoretically motivated learning rate scheduling in terms of $t$ and $p$ could be worthwile. 
- Improving approximation methods for the optimal $p$ computations would have runtime and stability benefits.

I guess I might be working on this in the following weeks.

### Cite this blog

If you found this small write up pertinent to your research, please consider citing.

```bibtex
@article{massena_pmuon,
  title={Generalizing Muon to Schatten-p norms},
  author={Thomas Massena},
  year={2026}
}
```

