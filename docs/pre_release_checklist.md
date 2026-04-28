# Pre-release checklist

- Generated Python caches are not included:

```bash
find . -name "__pycache__" -o -name "*.pyc"
```

The command should return no paths.
