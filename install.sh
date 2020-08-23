#!/usr/bin/env bash

ROOT=/home/hannes/work
URLBASE = https://github.com/astromancer
for name in recipes graphing motley obstools pySHOC; do
    git ls-remote ${name} -q
    if [[ $? -eq 0 ]]; then
        cd ${SRC}/${name}
        git pull
        cd ${SRC}
    else
        git clone ${URLBASE}/${name}
        git checkout dev
        python3 ${SRC}/${name}/setup.py
    fi
    done
