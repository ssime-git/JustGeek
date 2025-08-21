---
layout: default
title: Welcome to JustGeek
---

# Welcome to JustGeek

> A tech blog for geeks, by geeks

## Latest Posts

<div class="posts">
  {% for post in site.posts limit:5 %}
    <article class="post">
      <h3><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h3>
      <div class="date">{{ post.date | date: "%B %d, %Y" }}</div>
      <div class="excerpt">{{ post.excerpt }}</div>
    </article>
  {% endfor %}
</div>

---

## About This Blog

This blog covers:
- **Machine Learning** & AI insights
- **Programming** tutorials and tips  
- **Tech reviews** and analysis
- **Open source** projects
- **Development** best practices

Ready to dive deep into the world of technology? Let's get geeky! 🤓

## Recent Topics

Check out our latest posts above, or browse by category:
- [All Posts]({{ "/archive" | relative_url }})
- [About]({{ "/about" | relative_url }})