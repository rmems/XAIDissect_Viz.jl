#!/usr/bin/env bash
# Cursor Cloud Agent install script for XAIDissect_Viz.jl (`install` in .cursor/environment.json).
#
# Cursor runs this from the repository root during every Build, on its default
# Ubuntu base image (CPU only: cloud agents have no GPU), then snapshots the disk.
# It must be idempotent. Shell exports don't survive into agent runs, so the tools
# it installs are exposed through /etc/profile.d and /usr/local/bin.
# See https://cursor.com/docs/cloud-agent/setup
#
# Installs only what this repo's CI and manifests need:
#   - apt: xvfb, libgl1, mesa-utils, curl, ca-certificates, xauth
#   - juliaup 1.12.6 [default]
#   - Pkg.instantiate() + Pkg.precompile() under xvfb-run (as in ci.yml)
#   - tool directories exposed to later shells (/etc/profile.d + /usr/local/bin links)
#
# It ends with a dependency fetch/prebuild, not a test run.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  SUDO="sudo"
fi

# Install apt packages that are not already present.
apt_install() {
  local missing=() pkg
  for pkg in "$@"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null | grep -q "install ok installed"; then
      missing+=("$pkg")
    fi
  done
  if [ "${#missing[@]}" -gt 0 ]; then
    $SUDO apt-get -o Acquire::Retries=5 update -qq
    $SUDO env DEBIAN_FRONTEND=noninteractive apt-get -o Acquire::Retries=5 install -y --no-install-recommends "${missing[@]}"
  fi
}

# --- System packages (headless OpenGL for GLMakie, as in ci.yml; xauth, which xvfb-run needs and
# --no-install-recommends would skip; curl for the installers) ---
apt_install xvfb libgl1 mesa-utils curl ca-certificates xauth

# --- Julia (juliaup) ---
# ci.yml pins Julia 1.12.6.
JULIA_CHANNEL="1.12.6"
export PATH="$HOME/.juliaup/bin:$PATH"
if ! command -v juliaup >/dev/null 2>&1; then
  installer="$(mktemp)"
  trap 'rm -f "$installer"' EXIT
  curl -fsSL https://install.julialang.org -o "$installer"
  sh "$installer" --yes --default-channel "$JULIA_CHANNEL"
  rm -f "$installer"
  trap - EXIT
fi
if ! juliaup status | awk -v ch="$JULIA_CHANNEL" '{for (i = 1; i <= NF; i++) if ($i == ch) found = 1} END {exit !found}'; then
  juliaup add "$JULIA_CHANNEL"
fi
juliaup default "$JULIA_CHANNEL"

# --- Julia dependencies ---
# Same as ci.yml: bypass the pkg-server CDN and precompile after instantiate, under a
# headless X server because GLMakie needs a display.
for attempt in 1 2 3; do
  if JULIA_PKG_SERVER="" JULIA_PKG_PRECOMPILE_AUTO=0 \
    xvfb-run -a julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'; then
    break
  fi
  if [ "$attempt" -eq 3 ]; then
    exit 1
  fi
  echo "Instantiate/precompile attempt $attempt failed; retrying after cleanup" >&2
  rm -rf "$HOME/.julia/packages" "$HOME/.julia/artifacts" "$HOME/.julia/compiled"
  sleep 10
done

# --- Expose the tools to later shells ---
# The PATH exports above last only for this script; Cursor starts the agent's shells
# separately. Login shells get these directories from /etc/profile.d, and every other
# shell finds the entry points through symlinks in /usr/local/bin (on the default PATH).
tool_dirs=("$HOME/.juliaup/bin")
# shellcheck disable=SC2016 # $PATH must expand when the profile is sourced, not now.
printf 'export PATH="%s:$PATH"\n' "$(IFS=:; echo "${tool_dirs[*]}")" |
  $SUDO tee /etc/profile.d/cursor-env-XAIDissect_Viz.jl.sh >/dev/null
for dir in "${tool_dirs[@]}"; do
  [ -d "$dir" ] || continue
  for tool in "$dir"/*; do
    name="${tool##*/}"
    case "$name" in
      python* | pip* | activate* | deactivate | Activate.ps1) continue ;;
    esac
    if [ -f "$tool" ] && [ -x "$tool" ]; then
      target="/usr/local/bin/$name"
      if { [ -e "$target" ] || [ -L "$target" ]; } &&
        [ "$(readlink -f "$target")" != "$(readlink -f "$tool")" ]; then
        echo "Refusing to replace existing $target" >&2
        exit 1
      fi
      $SUDO ln -sfn "$tool" "$target"
    fi
  done
done

echo "Cursor install for XAIDissect_Viz.jl finished."
