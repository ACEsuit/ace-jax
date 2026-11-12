#!/usr/bin/env bash
# Publish a built site to the gh-pages branch (the repository's Pages source), or remove a preview.
#
#   docs/publish_pages.sh SITE_DIR            the main site: the branch root, previews under pr/ kept
#   docs/publish_pages.sh SITE_DIR pr/123     a pull request's preview, at <site>/pr/123/
#   docs/publish_pages.sh --remove pr/123     remove that preview
#
# The branch holds one commit, replaced on every publish (no history of built sites).  Two publishes may
# race: the push is conditional on the branch still being the commit that was read (--force-with-lease),
# and a loser starts again from the new branch.
# Previews carry <meta name="robots" content="noindex">, and nothing on the main site links to them.
# Needs GITHUB_TOKEN (contents: write) and GITHUB_REPOSITORY, or PAGES_REMOTE (any git URL; for tests).
set -euo pipefail
remove=; [ "${1:-}" = --remove ] && { remove=1; shift; }
if [ -n "$remove" ]; then site=; dest=${1:?pr/N}; else site=${1:?site directory}; dest=${2:-}; fi
[ -z "$dest" ] || [[ "$dest" =~ ^pr/[0-9]+$ ]] || { echo "destination must be pr/<number>, not '$dest'" >&2; exit 2; }
[ -z "$site" ] || [ -f "$site/index.html" ] || { echo "$site has no index.html" >&2; exit 2; }
url=${PAGES_REMOTE:-https://x-access-token:${GITHUB_TOKEN:-}@github.com/${GITHUB_REPOSITORY:-}.git}

attempt() {
  local work old
  work=$(mktemp -d)
  old=$(git ls-remote --heads "$url" gh-pages | cut -f1)
  if [ -n "$old" ]; then
    git clone -q --depth 1 --branch gh-pages "$url" "$work"
    old=$(git -C "$work" rev-parse HEAD)
  else
    git init -q -b gh-pages "$work"; git -C "$work" remote add origin "$url"
  fi
  if [ -n "$remove" ]; then
    [ -d "$work/$dest" ] || { echo "no preview at $dest"; return 0; }
    rm -rf "${work:?}/$dest"
  elif [ -n "$dest" ]; then
    mkdir -p "$work/$dest"
    rsync -a --delete "$site"/ "$work/$dest"/
    find "$work/$dest" -name '*.html' -exec sed -i '0,/<head>/s//<head><meta name="robots" content="noindex">/' {} +
  else
    rsync -a --delete --exclude /.git --exclude /pr "$site"/ "$work"/
  fi
  touch "$work/.nojekyll"
  git -C "$work" checkout -q --orphan publish
  git -C "$work" add -A
  git -C "$work" -c user.name="github-actions[bot]" -c user.email="41898282+github-actions[bot]@users.noreply.github.com" \
      commit -q -m "site: ${remove:+remove }${dest:-main} (${GITHUB_SHA:-local})"
  git -C "$work" push -q --force-with-lease=gh-pages:"$old" origin publish:gh-pages || return 1
  echo "published ${remove:+removal of }${dest:-the main site}; previews: $(ls "$work/pr" 2>/dev/null | tr '\n' ' ')"
}

for i in 1 2 3 4 5; do
  attempt && exit 0
  echo "gh-pages moved (attempt $i); starting again" >&2; sleep $((i * 3))
done
echo "could not publish after 5 attempts" >&2; exit 1
