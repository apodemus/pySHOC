#!/usr/bin/env bash

ROOT=~/Downloads
URL=https://github.com/astromancer
for name in recipes graphing motley obstools pySHOC; do
    git ls-remote ${name} -q > /dev/null 2>&1
    if [[ $? -eq 0 ]]; then
        cd ${ROOT}/${name}
        git pull
    else
        git clone ${URL}/${name}
        cd ${ROOT}/${name}
	    git checkout dev
        python3 setup.py
    fi
    echo
    cd ${ROOT}
    done