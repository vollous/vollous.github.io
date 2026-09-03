# vollous.github.io

Source for [vollous.github.io](https://vollous.github.io) — Chico Viana's
portfolio of data science and machine learning project write-ups.

## Structure

- `_notes/portfolio/` — the project write-ups (published at `/notes/<slug>`)
  and their `*-media/` image folders.
- `pages/` — home, about, projects index, 404, credits.
- `_plugins/obsidian_math.rb` — Obsidian-style `$…$` / `$$…$$` math.
- `_plugins/obsidian_embeds.rb` — Obsidian-style `![[image.png]]` embeds.
- `_layouts/`, `_includes/`, `assets/` — theme templates and static assets.

## Local development

```bash
bundle install
bundle exec jekyll serve
```

The site is built and deployed to GitHub Pages by
`.github/workflows/pages.yml` on every push to `main`.

## Credits

Built on the [Jekyll Garden](https://github.com/Jekyll-Garden/jekyll-garden.github.io)
theme (itself based on [Simply Jekyll](https://github.com/rgvr/simply-jekyll)).
Full acknowledgements on the [Credits](https://vollous.github.io/credits) page.
