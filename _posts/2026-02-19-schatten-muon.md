---
title: "Adaptive Schatten-p Norm Descent: Interpolating between SGD and Muon"
date: 2026-02-19 12:00:00 +0100
tags: [optimization]
categories: [blog]
layout: posts
---

# Using Schatten-p norms to dynamically interpolate between SGD and Muon
Thomas MASSENA | [Link to repository](https://github.com/massena-t/adaptive-ns)


<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #f9f9f9;">
<strong>TLDR</strong>: Using an adaptive scheme Newton-Schulz allows a theoretically principled way to choose between SGD-<em>esque</em> or Muon-<em>esque</em> weight updates.
</div>

### A Bit of Context

Since 2014-ish, the Adam optimizer has become the de-facto gradient-based optimization method for Deep Neural Networks (DNNs). Recently however, a concurrent optimizer named Muon has made headlines for its consistent gains over Adam in a variety of different scenarios. 

Muon uses the following update rule on 2D parameters after momentum accumulation:

$$
 W_{t+1} = W_t - \eta_t . \mathrm{PolarFactor}(M_t)
$$

with $M_t \in \mathbb{R}^{m,n}$ the update after momentum accumulation, of Singular Value Decomposition (SVD): $U \Sigma V^T$, with $U$ and $V$ orthogonal (or semi-orthogonal) matrices. The $\mathrm{PolarFactor}$ function returns the closest orthogonal matrix to $M_t$, i.e.: 

$$ 
 \mathrm{PolarFactor}(M_t) = UV^T
$$

Importantly, the computation of the Polar Factor can be efficiently approximated via a Newton-Schulz iterative scheme. And, as described in Bernstein and Newhouse's "Old Optimizer, New Norm" paper. Muon (and Shampoo), without E.M.A correspond to scaled solutions of the following steepest descent direction problem:

$$
 \mathrm{argmin}_{\Delta G \in \mathbb{R}^{m,n}} \langle G, \Delta W \rangle_F + \frac{\Lambda}{2} . \| \Delta W \|_2^2.
$$

meaning that the Muon update is a spectral descent direction under the maximum spectral norm update constraint (controlled by sharpness parameter $\Lambda$).

### Interpolating Between SGD and Muon

Now, given an unpreconditioned update $M_t$ of SVD $U \Sigma V^T$, let's propose the following update rule:

$$
 W_{t+1} = W_t - \eta_t . \left[ U \Sigma^{1/p} V^T \right], \forall \ p \in [1, \infty),
$$

which recovers the SGD update rule when $p=1$ and approaches Muon at exponential speed when $p \rightarrow \infty$. 


Using the "Old Optimizer, New Norm" framing, we find this update rule to be a steepest descent direction as it corresponds to the scaled version of solution to the following problem:

$$
 \mathrm{argmin}_{\Delta W \in \mathbb{R}^{m,n}} \langle G, \Delta W \rangle_F + \frac{\Lambda}{p+1}. \| \Delta W \|_{p+1}^{p+1}
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
\min_{s_i} \left( -\sigma_i s_i + \frac{\Lambda}{p+1} s_i^{p+1} \right)
$$

<p>Taking the derivative with respect to <em>s<sub>i</sub></em> and setting it to zero gives:</p>

$$
-\sigma_i + \Lambda s_i^p = 0 \implies s_i = \left(\frac{\sigma_i}{\Lambda}\right)^{1/p}
$$

<p>Absorbing the constant <em>(1/&Lambda;)<sup>1/p</sup></em> into the learning rate <em>&eta;<sub>t</sub></em>, we are left with the step's singular values <em>s<sub>i</sub> &prop; &sigma;<sub>i</sub><sup>1/p</sup></em>. Reconstructing the matrix with these new singular values yields the <em>U &Sigma;<sup>1/p</sup> V<sup>T</sup></em> update.</p>

</div>

### Efficient Approximation

Well, we now have a more general formulation of steepest descent directions that recovers both Muon and SGD. The main question now is: *can I compute $U \Sigma^{1/p} V^T$ efficiently ? How do we impose that in practice without using costly SVD computations ? The answer is actually quite easy. Newton-Schulz, again, for the win. Indeed, simply changing Newton-Schulz's coefficients suffices to approximate $U \Sigma^{1/p} V^T$ efficiently. 

<div style="border: 1px solid #ccc; padding: 10px; border-radius: 5px; background-color: #c1c1e1;">
Note: For more insights on Newton-Schulz coefficient computation, you can refer to <a href="https://leloykun.github.io/ponder/muon-opt-coeffs/" target="_blank">Franz Cesista's method</a> or even the <a href="https://arxiv.org/abs/2506.10935" target="_blank">Grishina (CANS)</a> or <a href="https://arxiv.org/abs/2505.16932" target="_blank">Amsel (Polar Express)</a> papers.
</div>

<br>

So, back to our endeavour, how do we compute these coeffs ? For the sake of you and I, let's keep it stupidly simple, we want to compute coefficients $\alpha$, $\beta$ and $\gamma$, such that:

$$
\begin{aligned}
 X_{n+1} &= a . X_n + b. X_n^3 + c . X_n^5 \\
\mathrm{lim}_{n \rightarrow \infty} X_{n} &= U \Sigma^{1/p} V^T
\end{aligned}
$$

with $X_0 = U \Sigma V^T$. That is, in terms of singular values: 

$$
 \min_{\alpha, \beta, \gamma \in \mathbb{R}^3} \int_0^1 (\alpha . x + \beta . x^3 + \gamma . x^5 - x^{1/p}) . dx \quad \mathrm{s.t} \quad \alpha + \beta + \gamma = 1.
$$

For better approximation performance, we can perform $N$ Newton-Schulz approximation steps and approximate $x_{t+1} = x_{t}^{p^{-1/N}}$ at each step. 

<div class="lagrange-system">
  <p>
    This is solved via the method of Lagrange multipliers, yielding a linear system
    \( A\mathbf{x} = \mathbf{b} \), where
    \( \mathbf{x} = [\alpha, \beta, \gamma, \lambda]^T \).
    The matrix entries \( A_{ij} \) correspond to the inner products of the basis functions
    \[
      \langle x^m, x^n \rangle
      = \int_0^1 x^{m+n} \, dx
      = \frac{1}{m+n+1}.
    \]
  </p>

  <p>The resulting system, which matches our implementation, is:</p>

  \[
  \begin{bmatrix}
  \frac{1}{3} & \frac{1}{5} & \frac{1}{7} & 1 \\
  \frac{1}{5} & \frac{1}{7} & \frac{1}{9} & 1 \\
  \frac{1}{7} & \frac{1}{9} & \frac{1}{11} & 1 \\
  1 & 1 & 1 & 0
  \end{bmatrix}
  \begin{bmatrix}
  \alpha \\ \beta \\ \gamma \\ \lambda
  \end{bmatrix}
  =
  \begin{bmatrix}
  \frac{1}{p + 2} \\
  \frac{1}{p + 4} \\
  \frac{1}{p + 6} \\
  1
  \end{bmatrix}.
  \]

  <p>
    The computation of these coefficients is fast given the small dimension of the problem.
    By using these coefficients, we approximate the Schatten-\(p\) update using only highly
    optimized matrix–matrix multiplications, reducing the computational cost per step.
  </p>
</div>


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

Nice ! We now have sort of a layerwise "*optimality*" condition on the value of $p$ for one step ! **There is a problem however, this computation requires knowing the full singular value spectra of** $A$ **and** $G$ **to find a solution via line-search algorithms**.

### Computing this Efficiently

Right now, anyone who has already trained big neural networks and ran into runtime or memory issues is probably cringing, knowing that the computation of the full singular value spectra of both the raw gradient and the post-activations is computational suicide.

<figure style="text-align: center;">
  <img src="/assets/images/pMuon/clenched-fist.png"
       style="max-width: 75%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.2em; color: #666;"> <em> Probably your initial reaction ? </em>
  </figcaption>
</figure>

Let's try to provide an efficient approximate of $p^*$. To that end, we need to compute a subset of singular values of $G$ and $A$. Therefore, we will either use a top-k approach to singular value decomposition, or a randomized subsampling strategy.

<div style="border: 1px solid #ccc; padding: 15px; border-radius: 5px; background-color: #f4f6f9; margin-top: 15px; margin-bottom: 20px;">
  <p>For massive weight matrices where even Randomized SVD is computationally prohibitive, a hybrid strategy isolates the spectral norm from the bulk distribution:</p>
  <ul>
    <li style="margin-bottom: 8px;"><strong>The Anchor:</strong> The objective $J(p)$ contains a singularity at $p=1$, where the denominator exponent diverges. To stabilize this regime, the exact spectral norm $\sigma_1$ is computed via Power Iteration, requiring only matrix-vector products ($O(k_{iter}\cdot d^2)$).</li>
    <li><strong>The Distribution:</strong> The remaining spectrum is estimated via Uniform Submatrix Sampling. A submatrix $M_{sub}\in\mathbb{R}^{m\times m}$ (where $m\ll d)$ is sampled, and its singular values are computed and scaled by $d/m$ to preserve spectral density.</li>
  </ul>
  <p style="margin-bottom: 0;">Stochastic subsampling introduces variance that can destabilize the trajectory of $p^*$. To mitigate this without increasing the sampling cost, Spectral Momentum is employed. Maintaining the exponential moving average (EMA) of spectral statistics over training steps dampens the noise from random sampling while allowing $p^*$ to adapt to the shifting geometry of the loss landscape.</p>
</div>

Indeed, several elements can mitigate the computational overhead of using our method. Without going into too much detail:
(i) we can run the $p$ updates every N steps (ii) we can run the computation of our optimal $p$ using subsampled $G$ and $A$ matrices (iii) we can group layers into sequential chunks where we will compute the mean optimal $p$ value to update coefficients. 
To counteract the potentially high variance of our singular value estimators, we use exponential moving average of the singular values across updates, then we compute the $p$ update.

Also, finally, our iterative scheme to update $p$ can be run on a separate process from the training run, which will update $p$ between training steps when a new $p$ value is recovered.

So to summarize, if we subsample the gradient and activation matrices of $L$ sequential chunks by a factor $S$, the extra overhead required by our algorithm is:
- **Runtime**: None if the $p$ update does not hold up the training process.
- **Memory**: requires storing a $L \times S \times \mathrm{min}(m,n)$ vector containing randomly sampled singular values.
- **Bandwidth**: In a multi-gpu setting the subsampled singular value vectors should probably be synchronised across devices.

In practice, if the optimal $p$ value is a relatively stable quantity, the fact that its update lags behind the training process should not be too much of an issue. 

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

Given that our optimizer is now adaptive, what insights do we recover about the optimal $p$ update across layers ? For example, on that same ResMLP network, we notice:

<figure style="text-align: center;">
  <img src="/assets/images/pMuon/optimal_p_values_resmlp.png"
       style="max-width: 100%; height: auto; display: block; margin: 0 auto;">
  <figcaption style="margin-top: 8px; font-size: 1.0em; color: #666;">
    <strong> The optimal p-value across layers can vary quite strongly along the depth of the ResMLP network, while staying stable. </strong>
  </figcaption>
</figure>

This recovers a typical empirical insight about the Muon optimizer, e.g., why people usually tend to exclude first and last layers from the Muon-optimized groups of parameters. Also, we notice that the optimal layerwise $p$ value tends to not be too unstable and that updating $p$ more often during early training and less often at the end might be a good strategy to gain a computational advantage.

### Limitations and Opportunities

While this whole study shows that Muon orthogonalization can be made adaptive. I believe the following elements could still be improved:
- The coefficient computation method could probably be improved via Chebyshev-type polynomial algorithms (see the Amsel et al. & Grishina et al. papers).
- Deriving the proper $\eta_{t,p}$ empirically or theoretically motivated learning rate scheduling in terms of $t$ and $p$ could be worthwhile. 
- Improving approximation methods for the optimal $p$ computations would have runtime and stability benefits.
- Adding second order momentum that depends on the optimal $p$ value ? 
- Investigating whether the optimal $p$ derived under the random feature regression framework remains well-suited when applied to the non-convex, multi-layer dynamics of deep neural network training is still an open question.

I might be working on this in the following weeks.

### Conclusion

While this whole theoretical derivation turned out pretty good. Further experimentation will be needed to determine whether the extra overhead this method introduces is worthwhile in the grand scheme of things, especially on the bitter lesson side of things. 
Anyhow, even if it is not worth it in terms of runtime / memory / training quality, I believe this method could be particularly interesting to understand the underlying dynamics of deep neural network training.

### Disclaimer

I am still relatively new to the optimization litterature. Please let me know if you would like to discuss any corrections, research directions and improvements for my work. 

### Cite this blog

If you found this small write up pertinent to your research, please consider citing.

```bibtex
@article{massena_pmuon,
  title={Generalizing Muon to Schatten-p norms},
  author={Thomas Massena},
  url={http://massena-t.github.io/blog/2026/02/19/schatten-muon.html},
  year={2026}
}
```

