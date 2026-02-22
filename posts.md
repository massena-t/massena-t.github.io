---
title: "Blogposts"
permalink: /posts/
layout: posts
---

# My Latest Blogposts

<ul>
{% for post in site.posts %}
  <li>
    <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
    <small> — {{ post.date | date: "%b %d, %Y" }}</small>
  </li>
{% endfor %}
</ul>

