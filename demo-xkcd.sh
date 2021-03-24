#!/bin/bash

# Should output:
# xkcd/1.png: 0.748037
# xkcd/2.png: 0.754339
# xkcd/3.png: 0.758869
# xkcd/4.png: 0.76831
# (and a lot of garbage)

rdmd filescore.d xkcd/*.png
