ROOT=/home/hannes/work
URLBASE = https://github.com/astromancer
for name in recipes graphing motley obstools pySHOC; do
    git clone $URLBASE/$name
    python3 $SRC/$name/setup.py
    done
