# Span Queries

What if we had a way to plan and optimize GenAI like we do for
[SQL](https://en.wikipedia.org/wiki/SQL)?

A **Span Query** is a declarative way to specify which portions of a
generative AI (GenAI) program should be **run directly on model
serving components**. As with
[SQL](https://en.wikipedia.org/wiki/SQL), this declarative structure
is safe to run on the backend and provides a clean starting point for
optimization. Also like SQL, some GenAI programs will be entirely
expressible as queries, though most will be expressed as the
programmatic interludes around the declarative queries.
