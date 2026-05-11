#!/usr/bin/env bash

set -e

VERSION=$1

if [ -z "$VERSION" ]; then
  echo "Usage: ./scripts/release.sh vX.Y.Z"
  exit 1
fi

git checkout main

git pull origin main

git tag -a "$VERSION" -m "Release $VERSION"

git push origin main

git push origin "$VERSION"

echo "Release $VERSION published"
