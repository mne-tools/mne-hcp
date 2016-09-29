#!/usr/bin/bash

(git co gh-pages &&
 cd .. &
 cp -R ./doc/_build/html/* . &&
 git add -f auto_examples/* &&
 git add -f auto_tutorials/* &&
 git add -f _images/* &&
 git add -f _sources/* &&
 git add -f _downloads/*)