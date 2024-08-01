#!/bin/bash

# Run flake8 with specified rules
flake8 . --count --max-complexity=13 --max-line-length=120 \
    --exclude='./Docs' \
    --per-file-ignores="__init__.py:F401" \
    --statistics