#!/bin/bash
# Rebuild Pion/ = upstream OPTML-Group/Pion @ 8796501 + our patches, on branch lang-spectral.
# Pion/ is not committed to this repo (it has its own git history); patches/ is the source of truth.
# To export new Pion commits:  git -C Pion format-patch <last-exported>..HEAD -o patches/
set -euo pipefail
cd "$(dirname "$0")/.."
if [ -d Pion/.git ]; then echo "Pion/ already exists:"; git -C Pion log --oneline -5; exit 0; fi
git clone -q https://github.com/OPTML-Group/Pion Pion
cd Pion
ID=(-c user.name=lang-vla -c user.email=lang-vla@users.noreply.github.com)
git checkout -q -b lang-spectral 8796501
git apply --index ../patches/pion-lang-spectral.patch
git "${ID[@]}" commit -qm "Apply lang_spectral patch"
git "${ID[@]}" am -q ../patches/0*.patch
git log --oneline -5
