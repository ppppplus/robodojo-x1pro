#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${repo_root}/x1pro/assets"
cache="${repo_root}/.cache/x1pro_assets_repo"

usage() {
  cat <<'HELP'
Usage: bash x1pro/init_assets.sh --from-dir /path/to/x1pro/assets
       bash x1pro/init_assets.sh --hf-repo OWNER/DATASET [--revision REV]

The model bundle is not included in the code repository. The Hugging Face
dataset must contain x1pro/assets/{fx001_h_evt1,rm001_g1_dvt1}/.
HELP
}

verify_bundle() {
  local root="$1" variant required
  for variant in fx001_h_evt1 rm001_g1_dvt1; do
    for required in kinematics.urdf appearance/x1pro_appearance.usd usd_control/ex001.usd usd_control/configuration/ex001_base.usd; do
      if [[ ! -f "${root}/${variant}/${required}" ]]; then
        echo "Missing model asset: ${root}/${variant}/${required}" >&2
        return 1
      fi
    done
    if [[ "$(head -c 8 "${root}/${variant}/usd_control/configuration/ex001_base.usd")" != 'PXR-USDC' ]]; then
      echo "Model USD is an LFS pointer, not a binary USD: ${variant}" >&2
      return 1
    fi
  done
}

if [[ $# -lt 2 ]]; then
  usage >&2
  exit 2
fi

mode="$1"
value="$2"
shift 2
revision=main
if [[ $# -gt 0 ]]; then
  if [[ $# -ne 2 || "$1" != '--revision' ]]; then
    usage >&2
    exit 2
  fi
  revision="$2"
fi

if [[ -e "$target" || -L "$target" ]]; then
  verify_bundle "$target"
  echo "X1 Pro model assets are ready: ${target}"
  exit 0
fi

case "$mode" in
  --from-dir)
    source_dir="$(realpath "$value")"
    verify_bundle "$source_dir"
    ln -s "$source_dir" "$target"
    ;;
  --hf-repo)
    if [[ ! "$value" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]]; then
      echo 'Expected Hugging Face dataset ID: OWNER/DATASET' >&2
      exit 2
    fi
    git lfs version >/dev/null || { echo 'git-lfs is required to download model assets' >&2; exit 1; }
    mkdir -p "$(dirname "$cache")"
    if [[ ! -d "$cache/.git" ]]; then
      GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 --sparse \
        "https://huggingface.co/datasets/${value}" "$cache"
    fi
    git -C "$cache" sparse-checkout set x1pro/assets
    git -C "$cache" fetch --depth 1 origin "$revision"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$cache" checkout FETCH_HEAD
    git -C "$cache" lfs install --local >/dev/null
    git -C "$cache" lfs pull --include='x1pro/assets/**' --exclude=''
    verify_bundle "$cache/x1pro/assets"
    ln -s "$cache/x1pro/assets" "$target"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

verify_bundle "$target"
echo "X1 Pro model assets are ready: ${target}"
