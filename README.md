hugo:https://gohugo.io/getting-started/quick-start/
hugo paper template: 


## using hugo
followed this setup doc https://gohugo.io/getting-started/quick-start/

## making changes
inside `content/posts/` at the top of the page there is something like this
```
+++
date = '2025-01-05T12:26:04-05:00'
draft = true
title = 'My First Post'
+++
```
👆 this heading makes whatever page you are working on a draft, you have to build with drafts to see your draft

```
hugo server --buildDrafts
hugo server -D
```

to release
```
hugo
```