# Spans on vLLM

> [!WARNING]
> To see the benefits of relocatable blocks currently requires a
> branch of vLLM. Stay tuned!

When using this branch of vLLM, you may launch the OpenAI-compatible
vLLM model serving endpoint with spans enabled:

```bash
VLLM_SPNL=True \
    VLLM_USE_V1=1 \
    VLLM_V1_SPANS_ENABLED=True
    VLLM_V1_SPANS_TOKEN=10
    VLLM_V1_SPANS_TOKEN_RECOMPUTE=31
    HF_TOKEN=...
    vllm serve ldsjmdy/Tulu3-Block-FT
```

## Direct REST Calls

To send a query with curl or any other REST-capable client, first prepare the query shape:

```bash
curl -s -XPOST http://localhost:8000/v1/query/prepare --data @./query.json -o /dev/null -w "%{time_total}\n"
1.504452
```

And then you can execute the query, and you should see millisecond-level TTFT:

```bash
curl -s -XPOST http://localhost:8000/v1/query/execute --data @./querya.json -o /dev/null -w "%{time_total}\n"
0.077699
```

## Using the SPNL Demo CLI

Coming soon.

