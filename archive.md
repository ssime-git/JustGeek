---
layout: default
title: Archive
permalink: /archive/
---

# All Posts

<div class="post-list">
  {% for post in site.posts %}
    <article class="post-item">
      <div class="post-meta">{{ post.date | date: "%B %d, %Y" }}</div>
      <h3 class="post-title">
        <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      </h3>
      {% if post.excerpt %}
        <div class="post-excerpt">{{ post.excerpt | strip_html | truncatewords: 30 }}</div>
      {% endif %}
    </article>
  {% endfor %}
</div>