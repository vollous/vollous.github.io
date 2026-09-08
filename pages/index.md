---
layout: Post
permalink: /
---

<div class="hero" markdown="1">

<img class="hero-avatar" src="/assets/img/profile.jpeg" alt="Chico Viana" />

# João "Chico" Viana

Data science & machine learning, with a physicist's touch.

During my PhD in theoretical physics, I spent most of my time developing theories and using computers to test them. Much of that is the same work data science asks for: building models, fitting them to data, and checking whether the fit means anything. I'm now moving that into data science and machine learning, an interest that has grown steadily over the past few years and that I wish to pursue.

<p class="hero-meta">
<span class="hero-meta-item"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/><circle cx="12" cy="10" r="3"/></svg>Lisbon, Portugal</span>
<a class="hero-meta-item" id="email-copy" href="mailto:jfvvchico@hotmail.com" data-email="jfvvchico@hotmail.com"><svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="m22 7-8.97 5.7a1.94 1.94 0 0 1-2.06 0L2 7"/></svg><span class="email-copy-label">jfvvchico@hotmail.com</span></a>
</p>

<span class="hero-actions">
<a class="btn btn-brand" href="{{ site.baseurl }}/projects/about-me">About me</a>
<a class="btn btn-ghost" href="{{ site.baseurl }}/assets/Curriculum_Vitae_Joao_Viana.pdf">Curriculum Vitae</a>
<a class="btn btn-ghost" href="https://github.com/vollous"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2 0 1.9 1.2 1.9 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2 0-.4-.5-1.6.2-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.6.2 2.8.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.5.3.9 1 .9 2.2v3.3c0 .3.1.7.8.6A12 12 0 0 0 12 .3"/></svg>GitHub</a>
<a class="btn btn-ghost" href="https://www.linkedin.com/in/jo%C3%A3o-viana-7aa4471a5/"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M20.45 20.45h-3.56v-5.57c0-1.33-.02-3.04-1.85-3.04-1.85 0-2.13 1.44-2.13 2.94v5.67H9.35V9h3.42v1.56h.05c.48-.9 1.64-1.85 3.37-1.85 3.6 0 4.27 2.37 4.27 5.46v6.28ZM5.34 7.43a2.07 2.07 0 1 1 0-4.14 2.07 2.07 0 0 1 0 4.14ZM7.12 20.45H3.55V9h3.57v11.45ZM22.22 0H1.77C.79 0 0 .77 0 1.72v20.56C0 23.23.79 24 1.77 24h20.45c.98 0 1.78-.77 1.78-1.72V1.72C24 .77 23.2 0 22.22 0Z"/></svg>LinkedIn</a>
</span>

</div>

<div class="section-head">
<h2 id="projects">Projects</h2>
<a class="btn btn-ghost" href="https://github.com/vollous/Portfolio"><svg viewBox="0 0 24 24" width="16" height="16" fill="currentColor" aria-hidden="true"><path d="M12 .3a12 12 0 0 0-3.8 23.4c.6.1.8-.3.8-.6v-2c-3.3.7-4-1.6-4-1.6-.6-1.4-1.4-1.8-1.4-1.8-1-.7.1-.7.1-.7 1.2 0 1.9 1.2 1.9 1.2 1 1.8 2.8 1.3 3.5 1 0-.8.4-1.3.7-1.6-2.7-.3-5.5-1.3-5.5-5.9 0-1.3.5-2.4 1.2-3.2 0-.4-.5-1.6.2-3.2 0 0 1-.3 3.3 1.2a11.5 11.5 0 0 1 6 0c2.3-1.5 3.3-1.2 3.3-1.2.7 1.6.2 2.8.1 3.2.8.8 1.2 1.9 1.2 3.2 0 4.6-2.8 5.6-5.5 5.9.5.3.9 1 .9 2.2v3.3c0 .3.1.7.8.6A12 12 0 0 0 12 .3"/></svg>Project code on GitHub</a>
</div>

<p class="section-intro">Each write-up goes through the whole project: looking at the data, trying out different models, and checking how well they actually work.</p>

<div class="project-card" markdown="1">
### [[Impact of AI on Students]]
A synthetic Kaggle dataset about students and AI use. I clean out values left over from the data being synthetic, then build several regression models to predict end-of-semester GPA (easy, it follows the starting GPA) and several classification models to flag burnout risk (hard), tuned to catch 90% of high-risk students at the cost of some accuracy.
<br>**Skills**: Python (NumPy, Matplotlib, pandas, PyTorch, scikit-learn).
</div>

<div class="project-card" markdown="1">
### [[Anomaly detection on the MVTec AD database]]
Spotting defective products after training only on good images (zero-shot), tested on
bottles, carpet, and hazelnuts. Convolutional autoencoders rebuild the image and
flag bad rebuilds, better architectures are needed. PatchCore needs no training,
compares each image to a memory bank of pretrained features, and was near
perfect, and it also shows where the defect is.
<br>**Skills**: Python (PyTorch, NumPy, Matplotlib), convolutional autoencoders, PatchCore, transfer learning, ROC/AUC evaluation.
</div>

<div class="project-card" markdown="1">
### [[RAG-Powered NumPy Documentation Assistant]]
A tiny 350M language model given a search layer over the NumPy docs, so it looks
up the right page before answering. Runs in three Docker containers
(llm, backend, frontend) via Docker Compose. The plain model hallucinates functions
and arguments. The version with RAG gets them right and admits when it can't.
<br>**Skills**: Python, LLMs, RAG (ChromaDB, Sentence Transformers), Ollama, FastAPI, Streamlit, Docker, Docker Compose, REST APIs.
</div>

<p class="more-link"><a href="{{ site.baseurl }}/projects">Browse all projects &rarr;</a></p>

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
