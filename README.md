Copy .msg and .srv files into the java package, and run rosgenjava.py -r

```
usage: rosgenjava.py [-h] [-t] [-a] [-s] [-r] filename

Process .srv/.msg file

positional arguments:
  filename

optional arguments:
  -h, --help       show this help message and exit
  -t, --tokenize   Run tokenizer only
  -a, --analyze    Run syntax analyze only
  -s, --show       Print generated java code only
  -r, --recursive  Walk beneath the path and process all .srv/.msg
```

ply is needed. Please install it in your python environment.

