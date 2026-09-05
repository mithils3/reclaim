#!/usr/bin/env bash
# Build the Apptainer image the reproduction agent runs every shell step inside.
# It layers git, curl, a C/C++ build chain, audio libs, Python 3.12 with headers, and uv
# onto a CUDA base image, because the sandbox runs --cleanenv --no-home and host modules
# never reach inside. Run where apt has network. Usage: bash build_cuda_sandbox.sh [BASE_SIF] [OUT_SIF]
set -euo pipefail

# <sif>: the CUDA + cuDNN development image to layer on top of.
BASE_SIF="${1:-<sif>}"
# <scratch>: a scratch/work root with room for the image and Apptainer's build cache.
OUT_SIF="${2:-<scratch>/cuda-agent.sif}"

export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-<scratch>/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-<scratch>/.apptainer/tmp}"
mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR"

DEF="$(mktemp --suffix=.def)"
trap 'rm -f "$DEF"' EXIT

cat > "$DEF" <<DEFEOF
Bootstrap: localimage
From: ${BASE_SIF}

%post
    set -eux
    export DEBIAN_FRONTEND=noninteractive
    apt-get update

    apt-get install -y --no-install-recommends \
        ca-certificates gnupg \
        git git-lfs \
        curl wget \
        unzip xz-utils \
        build-essential pkg-config cmake ninja-build \
        ffmpeg libsndfile1 sox \
        espeak-ng libespeak-ng1
    git lfs install --system || true

    # Python 3.12 with dev headers, so repos that build C/Cython extensions at install
    # time find Python.h at the conventional /usr/include/python3.12/.
    if ! apt-cache show python3.12 >/dev/null 2>&1; then
        apt-get install -y --no-install-recommends software-properties-common
        add-apt-repository -y ppa:deadsnakes/ppa
        apt-get update
    fi
    apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv python3-pip
    # Debian and Ubuntu disable ensurepip for the system interpreter, so pip comes from
    # the python3-pip package. Expose python3/python via /usr/local/bin without
    # repointing the distro's own /usr/bin/python3, which apt depends on.
    ln -sf /usr/bin/python3.12 /usr/local/bin/python3
    ln -sf /usr/bin/python3.12 /usr/local/bin/python

    # uv is baked in so the image is self-contained; a bound host uv overrides it.
    curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR=/usr/local/bin sh

    rm -rf /var/lib/apt/lists/*

%test
    set -e
    export PATH=/usr/local/bin:/usr/local/cuda/bin:\${PATH}
    for t in git git-lfs curl wget unzip gcc nvcc cmake ninja python3 uv espeak-ng; do
        command -v "\$t" >/dev/null && echo "ok: \$t -> \$(command -v \$t)" \
            || { echo "MISSING: \$t"; exit 1; }
    done
    echo "python: \$(python3 --version)"
    python3 -m pip --version
    inc="\$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"
    test -f "\$inc/Python.h" && echo "ok: Python.h -> \$inc/Python.h" \
        || { echo "MISSING: Python.h (no dev headers)"; exit 1; }
DEFEOF

echo ">> building ${OUT_SIF}"
echo ">>   from  ${BASE_SIF}"
# --fakeroot lets apt-get write into the image through user namespaces. Where it is
# disabled, build the same image somewhere you have root and copy the .sif over.
apptainer build --fakeroot "$OUT_SIF" "$DEF"

echo ">> verifying tools inside the image"
apptainer exec "$OUT_SIF" bash -lc \
  'export PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH; \
   which git git-lfs curl wget unzip gcc nvcc cmake ninja python3 uv espeak-ng; echo; \
   git --version; python3 --version; uv --version; espeak-ng --version | head -1'

echo ">> proving the Python dev toolchain can build+import a compiled extension"
apptainer exec "$OUT_SIF" bash -lc '
  set -e
  export PATH=/usr/local/bin:/usr/local/cuda/bin:$PATH
  tmp=$(mktemp -d)
  cat > "$tmp/ext.c" <<EOF
#include <Python.h>
static struct PyModuleDef mod = {PyModuleDef_HEAD_INIT, "ext", NULL, -1, NULL};
PyMODINIT_FUNC PyInit_ext(void) { return PyModule_Create(&mod); }
EOF
  inc=$(python3 -c "import sysconfig; print(sysconfig.get_path(\"include\"))")
  gcc -shared -fPIC -I"$inc" "$tmp/ext.c" -o "$tmp/ext.so"
  (cd "$tmp" && python3 -c "import ext; print(\"ok: built and imported a C extension\")")
  rm -rf "$tmp"'

echo ">> done: $OUT_SIF"
