---
layout: Post
permalink: /
---

<div class="hero" markdown="1">

<img class="hero-avatar" src="/assets/img/profile.jpeg" alt="Chico Viana" />

# João "Chico" Viana

Data science & machine learning, with a physicist's touch.

I spent my PhD teaching computers to describe the universe's first fractions of a second. Now I point the same curiosity at data problems closer to home. My goal is to design and train highly efficient models that run reliably on limited resources. 

<p class="hero-meta">
<span class="hero-meta-item"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>Lisbon, Portugal</span>
<a class="hero-meta-item" id="email-copy" href="mailto:jfvvchico@hotmail.com" data-email="jfvvchico@hotmail.com"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg><span class="email-copy-label">jfvvchico@hotmail.com</span></a>
</p>

<span class="hero-actions">
<a class="btn btn-brand" href="{{ site.baseurl }}/notes/curriculum-vitae">Curriculum Vitae</a>
<a class="btn btn-ghost" href="{{ site.baseurl }}/assets/Curriculum_Vitae_Joao_Viana.pdf">Download CV (PDF)</a>
<a class="btn btn-ghost" href="https://github.com/vollous"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2 0 1.9 1.2 1.9 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2 0-.4-.5-1.6.2-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.6.2 2.8.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.5.3.9 1 .9 2.2v3.3c0 .3.1.7.8.6A12 12 0 0 0 12 .3"/></svg>GitHub</a>
</span>

</div>

<div class="section-head">
<h2 id="projects">Projects</h2>
<a class="btn btn-ghost" href="https://github.com/vollous/Portfolio"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2 0 1.9 1.2 1.9 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2 0-.4-.5-1.6.2-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.6.2 2.8.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.5.3.9 1 .9 2.2v3.3c0 .3.1.7.8.6A12 12 0 0 0 12 .3"/></svg>Project code on GitHub</a>
</div>

<p class="section-intro">Each write-up covers the full path from exploratory analysis through model selection to evaluation.</p>

<div class="project-card" markdown="1">
### [[Impact of AI on Students]]
EDA, regression on post-semester GPA, and threshold-tuned classification of
burnout risk on a synthetic Kaggle dataset.
</div>

<div class="project-card" markdown="1">
### [[Anomaly detection on the MVTec AD database]]
Zero-shot defect detection comparing convolutional autoencoders against
PatchCore.
</div>

<div class="project-card" markdown="1">
### [[RAG-Powered NumPy Documentation Assistant]]
A sub-1B-parameter LLM boosted with a ChromaDB retrieval layer over the NumPy
documentation, deployed with Docker Compose.
</div>

<p class="more-link"><a href="{{ site.baseurl }}/notes">Browse all projects &rarr;</a></p>

<script>
(function () {
  var link = document.getElementById('email-copy');
  if (!link) return;
  var label = link.querySelector('.email-copy-label');
  var email = link.dataset.email;
  var original = label.textContent;
  var timer;
  link.addEventListener('click', function (e) {
    e.preventDefault();
    var done = function () {
      label.textContent = 'Copied!';
      clearTimeout(timer);
      timer = setTimeout(function () { label.textContent = original; }, 2000);
    };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(email).then(done).catch(done);
    } else {
      var t = document.createElement('textarea');
      t.value = email;
      document.body.appendChild(t);
      t.select();
      try { document.execCommand('copy'); } catch (err) {}
      document.body.removeChild(t);
      done();
    }
  });
})();
</script>
