# Details of Execution

The lifecycle of a span query includes:

- Input from client to the generate
- Output to client of the generate
- By-products: what the model server caches as a result of that generate

## Input Concerns

Input from client to the generate.

### Messages

```
(system m): a message with role "system" and content m
(user m): a message with role "user" and content m
(assistant m): a message with role "assistant" and content m
```

### Terminology

The terminology below has capitalized letters representing strings and
lowercase letters representing token sequences. We assume that when
mapping `A` to `a` the chat template is applied then the tokenizer.

```
A, B, C: these represent messages
a, b, c: these represent corresponding token sequences, with chat template applied
_: ensure that the preceding sequence both starts and ends on a block boundary
+: special token for begin span
x: special token for restore cross attention
```

### Rules

```
(seq A B C) -> abc
(plus A B C) -> (+a)_(+b)_(+c)_   meaning add + to each and ensure each starts and ends on a block boundary
(cross A B C) -> ab(xc)_          meaning add x before the last element and ensure (xc) starts and ends on a block boundary
```

### Examples

```
(cross A (plus B C) D) -> a(+b)_(+c)_(xd)_
```

## By-product of generate

What the model server caches as a result of that generate.

TODO
