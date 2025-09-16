---
title: "Home - massena-t"
---

<div style="display:flex;align-items:center;gap:1rem;flex-wrap:wrap">
  <img src="{{ "/assets/images/profile.jpg" | relative_url }}" alt="Portrait of Thomas Massena"
       style="width:140px;height:140px;object-fit:cover;border-radius:50%;border:1px solid #ddd;">
  <div>
    <h1 style="margin:0 0 .25rem 0;">About Me</h1>
    <p style="margin:.25rem 0;">
      I am currently a PhD student at IRIT and SNCF, working on certifiably robust deep learning.
    </p>

    <p style="margin:.5rem 0; display:flex; gap:.5rem; flex-wrap:wrap;">
      <a href="https://scholar.google.com/citations?user=n09aacYAAAAJ"
         target="_blank" rel="noopener noreferrer"
         style="display:inline-block;padding:.5rem .75rem;border:1px solid #ccc;border-radius:8px;text-decoration:none;">
        Google Scholar ↗
      </a>

      <a href="{{ site.github.owner_url | default: 'https://github.com/massena-t' }}"
         target="_blank" rel="noopener noreferrer"
         style="display:inline-block;padding:.5rem .75rem;border:1px solid #ccc;border-radius:8px;text-decoration:none;">
        GitHub ↗
      </a>
    </p>
  </div>
</div>

<hr style="margin:1.5rem 0;">

### News

<div style="border:1px solid #ddd;border-radius:10px;padding:1rem;">
  <p>
    I am honored to announce I received the <strong>Alexey Chervonenkis award</strong> for the 
    <strong>Best Poster</strong> at the Fourteenth Symposium on Conformal and Probabilistic Prediction 
    with Applications (COPA 2025).
  </p>
  <p>
    I was fortunate to share this honor with my brilliant coworker <strong>Léo Andéol</strong>.
  </p>
</div>


### Publications

<div style="display:flex;flex-direction:column;gap:1rem;">

  <!-- Publication 1 -->
  <div style="border:1px solid #ddd;border-radius:10px;padding:1rem;">
    <h4 style="margin:0 0 .25rem 0;">
      Efficient Robust Conformal Prediction via Lipschitz-Bounded Networks. (ICML 2025)
    </h4>
    <p style="margin:.25rem 0;color:#555;">
      <em>T. Massena*</em>, L. Andéol*, T. Boissin, F. Mamalet, C. Friedrich, M. Serrurier, S. Gerchinovitz
    </p>
    <p style="margin:.25rem 0;">
      We provide a method to enable ≈1000× more memory efficient Robust Conformal Prediction compared to related works without any performance loss. 
      This enables efficient prediction with guaranteed error rates in noisy or adversarial environments.
    </p>
    <p style="margin:.5rem 0 0 0;">
      <a href="https://arxiv.org/abs/2506.05434" target="_blank" style="text-decoration:none;color:#0366d6;">[Paper link.]</a>
    </p>
  </div>

  <!-- Publication 2 -->
  <div style="border:1px solid #ddd;border-radius:10px;padding:1rem;">
    <h4 style="margin:0 0 .25rem 0;">
      DP-SGD without Clipping: The Lipschitz Neural Network Way. (ICLR 2024)
    </h4>
    <p style="margin:.25rem 0;color:#555;">
      L. Béthune*, <em>T. Massena*</em>, T. Boissin*, A. Bellet, F. Mamalet, Y. Prudent, C. Friedrich, M. Serrurier, D. Vigouroux
    </p>
    <p style="margin:.25rem 0;">
      We show that Lipschitz-constrained neural networks allow fast and intuitive Differentially Private training, this reduces DP-SGD training time significantly and effectively eliminates detrimental clipping bias.
    </p>
    <p style="margin:.25rem 0;font-style:italic;">
      Also presented as an invited Google Tech Talk.
    </p>
    <p style="margin:.5rem 0 0 0;">
      <a href="https://arxiv.org/abs/2305.16202" target="_blank" style="text-decoration:none;color:#0366d6;">[Paper link.]</a>
    </p>
  </div>

  <!-- Publication 3 -->
  <div style="border:1px solid #ddd;border-radius:10px;padding:1rem;">
    <h4 style="margin:0 0 .25rem 0;">
      An Adaptive Orthogonal Convolution Scheme for Efficient and Flexible CNN Architectures. (ICML 2025)
    </h4>
    <p style="margin:.25rem 0;color:#555;">
      T. Boissin*, F. Mamalet, T. Fel, A. M. Picard, <em>T. Massena</em>, M. Serrurier
    </p>
    <p style="margin:.25rem 0;">
      We implement fast and flexible parametrizations for orthogonally constrained convolutions that match the time and memory consumption of unconstrained convolutions in large batch size settings.
    </p>
    <p style="margin:.5rem 0 0 0;">
      <a href="https://arxiv.org/abs/2501.07930" target="_blank" style="text-decoration:none;color:#0366d6;">[Paper link.]</a>
    </p>
  </div>

  <!-- Publication 4 -->
  <div style="border:1px solid #ddd;border-radius:10px;padding:1rem;">
    <h4 style="margin:0 0 .25rem 0;">
      Sequential Conformal Risk Control for Safe Railway Signaling Detection. (COPA 2025, Extended Abstract)
    </h4>
    <p style="margin:.25rem 0;color:#555;">
      L. Andéol, <em>T. Massena</em>
    </p>
    <p style="margin:.25rem 0;">
      We enable safe railway signaling detection risk control guarantees on detection confidence, localization, and classification.
    </p>
    <p style="margin:.5rem 0 0 0;">
      <a href="https://raw.githubusercontent.com/mlresearch/v266/main/assets/andeol25a/andeol25a.pdf" target="_blank" style="text-decoration:none;color:#0366d6;">[Paper link.]</a>
    </p>
  </div>

</div>


### Disclaimer

This page is currently being built, I will try to upload some interesting research snippets in the coming weeks.

<!-- - [/about/](/about/) -->
<!-- - [/posts/](/posts/) -->

