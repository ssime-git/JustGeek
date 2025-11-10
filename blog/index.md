---
layout: default
title: Blog
description: Deep dives, tutorials, and updates from JustGeek.
---

# Blog

Welcome to the complete archive. Browse the latest stories below or jump to a category.

## Latest Articles

<div class="post-list">
  {% for post in paginator.posts %}
    <article class="post-item">
      <div class="post-meta">{{ post.date | date: "%B %d, %Y" }}</div>
      <h2 class="post-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <div class="post-excerpt">{{ post.excerpt | strip_html | truncate: 200 }}</div>
      <a href="{{ post.url | relative_url }}" class="read-more">Read more →</a>
    </article>
  {% endfor %}
</div>

<nav class="pagination" role="navigation" aria-label="Pagination">
  {% if paginator.previous_page %}
    <a class="newer" href="{{ paginator.previous_page_path | relative_url }}">← Newer Posts</a>
  {% else %}
    <span class="disabled">← Newer Posts</span>
  {% endif %}

  <span class="page-count">Page {{ paginator.page }} of {{ paginator.total_pages }}</span>

  {% if paginator.next_page %}
    <a class="older" href="{{ paginator.next_page_path | relative_url }}">Older Posts →</a>
  {% else %}
    <span class="disabled">Older Posts →</span>
  {% endif %}
</nav>

## Browse by Category

<ul class="category-list">
  {% assign sorted_cats = site.categories | sort %}
  {% for category in sorted_cats %}
    <li>
      <a href="{{ '/categories/' | append: category[0] | slugify | append: '/' | relative_url }}">{{ category[0] }} ({{ category[1].size }})</a>
    </li>
  {% endfor %}
</ul>
