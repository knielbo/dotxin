#!/usr/bin/env bash
pdflatex main.tex
pdflatex main.tex
rm *.aux *.log *.out *.snm *.toc *.nav
xdg-open main.pdf
cp main.pdf /home/knielbo/Documents/knielbo.github.io/files/kln_shangshu.pdf
