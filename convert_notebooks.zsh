#!/bin/zsh

for ipynb in *ipynb
do
	echo "Converting $ipynb"
	jupyter nbconvert --to markdown $ipynb
done
