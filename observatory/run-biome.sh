#!/bin/bash
cd "$(dirname "$0")"
npx biome check --write src/ 2>&1
echo "SRC_EXIT=$?"
npx biome check --write dev/app.tsx dev/main.tsx 2>&1
echo "DEV_EXIT=$?"
npx biome check --write tests/ 2>&1
echo "TESTS_EXIT=$?"
