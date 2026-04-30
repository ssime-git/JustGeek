---
layout: default
title: Presentations
permalink: /presentations/
---

# Presentations

Talks, workshops, and slide decks published from the blog.

<div class="presentation-list">
  {% for presentation in site.data.presentations %}
  <article class="presentation-card">
    <div class="presentation-card__meta">
      <span>{{ presentation.mode | replace: "-", " " }}</span>
      <span>{{ presentation.slides }} slides</span>
      <span>{{ presentation.date | date: "%Y-%m-%d" }}</span>
    </div>
    <h2>{{ presentation.title }}</h2>
    <p>{{ presentation.description }}</p>
    <div class="presentation-card__tags">
      {% for tag in presentation.tags %}
      <span>{{ tag }}</span>
      {% endfor %}
    </div>
    <a class="presentation-card__link" href="{{ presentation.url | relative_url }}">Open slides</a>
  </article>
  {% endfor %}
</div>
