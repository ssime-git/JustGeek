---
layout: default
title: Blog
description: Deep dives, tutorials, and updates from JustGeek.
---

# Blog

Welcome to the complete archive. Browse the latest stories below or jump to a category.

## Latest Articles

<div class="post-list">
  {% for post in site.posts %}
    <article class="post-item">
      <div class="post-meta">{{ post.date | date: "%B %d, %Y" }}</div>
      <h2 class="post-title"><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <div class="post-excerpt">{{ post.excerpt | strip_html | truncate: 200 }}</div>
      <a href="{{ post.url | relative_url }}" class="read-more">Read more →</a>
    </article>
  {% endfor %}
</div>

## Browse by Category

<ul class="category-list">
  {% assign sorted_cats = site.categories | sort %}
  {% for category in sorted_cats %}
    <li>
      <a href="{{ '/categories/' | append: category[0] | slugify | append: '/' | relative_url }}">{{ category[0] }} ({{ category[1].size }})</a>
    </li>
  {% endfor %}
</ul>
