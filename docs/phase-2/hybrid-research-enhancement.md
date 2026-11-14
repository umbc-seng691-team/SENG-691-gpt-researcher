# Hybrid Local and Web Research

## What changed

Hybrid mode now loads local files and searches the web at the same time. Before this change, the web search did not start until the local files had finished loading.

The results from both sources are converted to text and joined before the report is written. Local and web content remain labeled separately so it is clear where the information came from.

## Error handling

The two searches are handled separately. If a local file cannot be read, the report can still use the web results. If the web search fails, the report can still use the local files. The run returns an empty result only when neither source provides usable content.

## Tests

The tests cover the following cases:

- Web research starts while local files are loading.
- Web results are kept when local-file loading fails.
- Local results are kept when web research fails.
- No context is returned when both searches fail.
- List results are converted to clean text before they are joined.

Run the tests with:

```bash
python -m pytest tests/skills/test_hybrid_research.py -v
```

## Contributor

Michael Obeng Antwi
