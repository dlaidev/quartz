#!/bin/bash
filename="content/$(date +%Y-%m-%d)-${1// /-}.md"
cat > "$filename" << EOF
---
title: "$1"
date: $(date +%Y-%m-%d)
tags: [AI, thoughts, compiler, inference, cuda, cute, cutlass, mlir, pytorch, jax]
---

# $1

EOF
code "$filename"  # or your editor


#!/bin/bash
cp templates/post-template.md "content/$(date +%Y-%m-%d)-${1// /-}.md"
sed -i '' "s/TITLE_PLACEHOLDER/$1/g" "content/$(date +%Y-%m-%d)-${1// /-}.md"
sed -i '' "s/DATE_PLACEHOLDER/$(date +%Y-%m-%d)/g" "content/$(date +%Y-%m-%d)-${1// /-}.md"
code "content/$(date +%Y-%m-%d)-${1// /-}.md"
