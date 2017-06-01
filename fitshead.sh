#!/usr/bin/env bash
head -c `head -c 10000 $1 | grep ' END' -aob | grep -oE '[0-9]+'` $1