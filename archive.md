---
layout: default
title: Archive
permalink: /archive/
---

# All Posts

<div class="posts-archive">
  {% for post in site.posts %}
    <div class="post-item">
      <span class="post-date">{{ post.date | date: "%Y-%m-%d" }}</span>
      <a href="{{ post.url | relative_url }}" class="post-link">{{ post.title }}</a>
    </div>
  {% endfor %}
</div>

<style>
.posts-archive {
  margin-top: 2rem;
}

.post-item {
  margin-bottom: 1rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid #333;
}

.post-date {
  color: #666;
  font-family: monospace;
  margin-right: 1rem;
}

.post-link {
  color: #00ff00;
  text-decoration: none;
}

.post-link:hover {
  text-decoration: underline;
}
</style>