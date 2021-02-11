
# Contributing


## Ensuring code quality

### Style checking

Style checks with yapf can be run using 

```bash
python tools/style_check.py --path=flatland
```

This tool requires the `yapf` tool to be available on your system path which can be accomplished through some variant of the following:

```bash
pip install yapf && export PATH=$PATH:$HOME/.local/bin
```
