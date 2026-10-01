---
layout: default
title: "Thomas Massena"
---

<div class="news-card">
  <p>As I approach the final year of my PhD, I am open to discussing new opportunities as an <strong>AI Researcher</strong> or <strong>AI Software Engineer</strong>.</p>
  <p>Feel free to reach out at <a href="mailto:thomasmassena@gmail.com">thomasmassena@gmail.com</a>.</p>
</div>

## News

<ul class="news-list">
  <li>
    <time datetime="2026-09-25">September 25, 2026</time>
    <p>Our paper, <em>From SGD to Muon: Adaptive Optimization via Schatten-p Norms</em>, was accepted at <strong>NeurIPS 2026</strong>.</p>
  </li>
  <li>
    <time datetime="2026-07-23">July 23, 2026</time>
    <p>Our paper, <em>Fast and Flexible Robustness Certificates for Semantic Segmentation</em>, was accepted at <strong>ECCV 2026</strong>.</p>
  </li>
  <li>
    <time datetime="2026-02-22">February 22, 2026</time>
    <p>Merged a PR on the <a href="https://github.com/google-deepmind/optax/pull/1602" target="_blank">Optax</a> repository containing coefficient presets and various normalization schemes for the Muon optimizer, along with the Turbo-Muon contributions.</p>
  </li>
</ul>

## Recorded Talks

I was invited to pitch my recent paper, "Fast and Flexible Robustness Certificates for Semantic Segmentation," at ANITI Days 2026 in Toulouse. You can watch the <a href="https://www.youtube.com/watch?v=YsUPKgzPnIA" target="_blank">ANITI Days talk</a> and the associated <a href="https://www.youtube.com/watch?v=KxZP6ML0T2s" target="_blank">five-minute ECCV presentation</a>.

## Awards

Together with my coworker <strong>Leo Andeol</strong>, I received the <strong>Alexey Chervonenkis Award for Best Poster</strong> at the Fourteenth Symposium on Conformal and Probabilistic Prediction with Applications (COPA 2025).

## Publications

<div class="publications">

  <div class="pub-card">
    <h4>From SGD to Muon: Adaptive Optimization via Schatten-p Norms <span class="pub-venue">NeurIPS 2026</span></h4>
    <p class="pub-authors"><em>T. Massena*</em>, C. Friedrich, M. Serrurier</p>
    <p class="pub-desc">Initially developed in this <a href="{% post_url 2026-02-19-schatten-muon %}">blog post</a>, we show that a first-order model of the LMO update rule's optimality is sufficient to match or improve upon the better-performing optimizer between Adam or Muon across diverse training tasks.</p>
    <a href="https://arxiv.org/abs/2605.19781" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

  <div class="pub-card">
    <h4>Turbo-Muon: Almost-Orthogonal Pre-Conditioning for Fast Muon Updates <span class="pub-venue">Pre-print, under review</span></h4>
    <p class="pub-authors">T. Boissin*, <em>T. Massena*</em>, F. Mamalet, M. Serrurier</p>
    <p class="pub-desc">We improve the efficiency of the costly Newton-Schulz iteration of the Muon optimizer by using a preconditioning method from the Approximately Orthogonal Layer paper from Prach et al. This allows us to conserve the impressive performance of the Muon optimizer while gaining substantial computational efficiency at scale.</p>
    <a href="https://arxiv.org/abs/2512.04632" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

  <div class="pub-card">
    <h4>Fast and Flexible Robustness Certificates for Semantic Segmentation <span class="pub-venue">ECCV 2026</span></h4>
    <p class="pub-authors"><em>T. Massena*</em>, C. Friedrich, F. Mamalet, M. Serrurier</p>
    <p class="pub-desc">We use Lipschitz neural networks to perform certifiably robust segmentation tasks on challenging datasets such as CityScapes. Our networks are approximately 600 to 2000 times more computationally efficient at inference time. We additionally develop a full framework for the certification of complex deep learning models under arbitrary threats under two different paradigms.</p>
    <a href="https://arxiv.org/abs/2512.06010" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

  <div class="pub-card">
    <h4>Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks <span class="pub-venue">ICML 2025</span></h4>
    <p class="pub-authors"><em>T. Massena*</em>, L. Andeol*, T. Boissin, F. Mamalet, C. Friedrich, M. Serrurier, S. Gerchinovitz</p>
    <p class="pub-desc">We provide a method to enable ~1000x more memory efficient Robust Conformal Prediction compared to related works without any performance loss. This enables efficient prediction with guaranteed error rates in noisy or adversarial environments.</p>
    <a href="https://arxiv.org/abs/2506.05434" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

  <div class="pub-card">
    <h4>DP-SGD without Clipping: The Lipschitz Neural Network Way <span class="pub-venue">ICLR 2024</span></h4>
    <p class="pub-authors">L. Bethune*, <em>T. Massena*</em>, T. Boissin*, A. Bellet, F. Mamalet, Y. Prudent, C. Friedrich, M. Serrurier, D. Vigouroux</p>
    <p class="pub-desc">We show that Lipschitz-constrained neural networks allow fast and intuitive Differentially Private training, reducing DP-SGD training time significantly and eliminating detrimental clipping bias.</p>
    <p class="pub-note">Also presented as an invited Google Tech Talk.</p>
    <a href="https://arxiv.org/abs/2305.16202" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

  <div class="pub-card">
    <h4>An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures <span class="pub-venue">ICML 2025</span></h4>
    <p class="pub-authors">T. Boissin*, F. Mamalet, T. Fel, A. M. Picard, <em>T. Massena</em>, M. Serrurier</p>
    <p class="pub-desc">We implement fast and flexible parametrizations for orthogonally constrained convolutions that match the time and memory consumption of unconstrained convolutions in large batch size settings.</p>
    <a href="https://arxiv.org/abs/2501.07930" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

  <div class="pub-card">
    <h4>Sequential Conformal Risk Control for Safe Railway Signaling Detection <span class="pub-venue">COPA 2025</span></h4>
    <p class="pub-authors">L. Andeol, <em>T. Massena</em></p>
    <p class="pub-desc">We enable safe railway signaling detection with risk control guarantees on detection confidence, localization, and classification.</p>
    <a href="https://raw.githubusercontent.com/mlresearch/v266/main/assets/andeol25a/andeol25a.pdf" target="_blank" class="pub-link">Paper &rarr;</a>
  </div>

</div>
