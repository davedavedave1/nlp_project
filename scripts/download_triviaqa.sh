#!/usr/bin/env bash
set -e

mkdir -p data
cd data

# TriviaQA download links
wget https://nlp.cs.washington.edu/triviaqa/data/triviaqa-rc.tar.gz

tar -xzf triviaqa-rc.tar.gz