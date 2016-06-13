#!/usr/bin/env bash
git reset --hard origin/gh-pages
git rebase origin/master
cd ..
python setup.py develop
cd doc
make clean && make html
git add build/html/*.html
git add build/html/generated/*.html
git add build/html/_sources/*.txt
git add build/html/_sources/generated/*.txt
git ls-files build | grep '\.html$' | xargs git add
git ls-files build | grep '\.txt$' | xargs git add
git ls-files build | grep '\.js$' | xargs git add
git ls-files build | grep '\.css$' | xargs git add
git add build/html/searchindex.js
git add build/html/_static/*
git diff --name-only
git commit --amend -C HEAD
git push -f
