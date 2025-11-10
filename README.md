# JustGeek
My Personal blog about Geek stuff

## Local development

Preview the site locally with Docker (no Ruby setup required):

```bash
docker run --rm -it \
  -v "$PWD":/srv/jekyll \
  -p 4000:4000 \
  jekyll/jekyll:4.3.2 \
  jekyll serve --livereload --host 0.0.0.0
```

Then open <http://localhost:4000> in your browser.
