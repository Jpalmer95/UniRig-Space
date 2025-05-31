#!/bin/bash
set -e

# --- Configuration ---
BLENDER_VERSION="4.2.0"
BLENDER_MAJOR_MINOR="4.2" # Corresponds to Blender's internal versioned Python directory
BLENDER_PYTHON_VERSION="python3.11" # Should match the -dev package (e.g., python3.11-dev)
BLENDER_TARBALL="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender${BLENDER_MAJOR_MINOR}/blender-${BLENDER_VERSION}-linux-x64.tar.xz"

APP_DIR="/home/user/app"
BLENDER_INSTALL_BASE="${APP_DIR}/blender_installation"
INSTALL_DIR="${BLENDER_INSTALL_BASE}/blender-${BLENDER_VERSION}-linux-x64"
LOCAL_BIN_DIR="${APP_DIR}/local_bin"

BLENDER_PY_EXEC="${INSTALL_DIR}/${BLENDER_MAJOR_MINOR}/python/bin/${BLENDER_PYTHON_VERSION}"

UNIRIG_REQS_FILE_IN_SPACE="${APP_DIR}/unirig_requirements.txt"
UNIRIG_REPO_CLONE_DIR="${APP_DIR}/UniRig"

TORCH_VERSION="2.3.1"
TORCHVISION_VERSION="0.18.1"
# ** MODIFIED: Changed PyTorch target to CUDA 11.8 to align with flash-attn wheel **
TARGET_CUDA_VERSION_SHORT="cu118"
TORCH_INDEX_URL="https://download.pytorch.org/whl/${TARGET_CUDA_VERSION_SHORT}"

# Direct URL for the compatible flash-attn wheel for v2.5.8
# Compatible with: Python 3.11 (cp311), PyTorch 2.3.x (torch2.3), CUDA 11.8 (cu118), CXX11 ABI TRUE
FLASH_ATTN_WHEEL_URL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu118torch2.3cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# --- Set Environment Variables for Build ---
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda} # This might be nominal if nvcc isn't actually used
export PATH="${CUDA_HOME}/bin:${LOCAL_BIN_DIR}:${PATH}"
export MAX_JOBS=${MAX_JOBS:-4} # For compilation jobs if any occur

PYTHON_INCLUDE_DIR="/usr/include/python${BLENDER_PYTHON_VERSION#python}"
export CPATH="${PYTHON_INCLUDE_DIR}:${CPATH}"
export C_INCLUDE_PATH="${PYTHON_INCLUDE_DIR}:${C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="${PYTHON_INCLUDE_DIR}:${CPLUS_INCLUDE_PATH}"

# TORCH_CUDA_ARCH_LIST is important if flash-attn *does* try to compile parts of itself.
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0" # Added older architectures for cu118 compatibility

echo "--- Setup Script Start ---"
echo "Target Blender Installation Directory: ${INSTALL_DIR}"
echo "Blender Python Executable: ${BLENDER_PY_EXEC}"
echo "Using CUDA_HOME=${CUDA_HOME}"
echo "Targeting PyTorch for CUDA: ${TARGET_CUDA_VERSION_SHORT}"
echo "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}"
echo "Attempting to install flash-attn from direct wheel URL: ${FLASH_ATTN_WHEEL_URL}"

# --- Download and Extract Blender ---
mkdir -p "${BLENDER_INSTALL_BASE}"
mkdir -p "${LOCAL_BIN_DIR}"

if [ ! -d "${INSTALL_DIR}" ] || [ -z "$(ls -A "${INSTALL_DIR}")" ]; then
    echo "Blender not found or directory empty at ${INSTALL_DIR}. Proceeding with download and extraction."
    echo "Downloading Blender ${BLENDER_VERSION}..."
    if [ ! -f "/tmp/${BLENDER_TARBALL}" ]; then
        wget -nv -O "/tmp/${BLENDER_TARBALL}" ${BLENDER_URL}
    else
        echo "Blender tarball /tmp/${BLENDER_TARBALL} already downloaded."
    fi

    echo "Extracting Blender to ${BLENDER_INSTALL_BASE}..."
    tar -xJf "/tmp/${BLENDER_TARBALL}" -C "${BLENDER_INSTALL_BASE}"
    if [ -d "${INSTALL_DIR}" ]; then
        echo "Blender extracted successfully to ${INSTALL_DIR}"
    else
        echo "ERROR: Blender extraction failed. Expected: ${INSTALL_DIR}"
        ls -la "${BLENDER_INSTALL_BASE}"
        exit 1
    fi
else
    echo "Blender already appears to be extracted to ${INSTALL_DIR}."
fi
echo "Extraction complete."

if [ -f "${INSTALL_DIR}/blender" ]; then
    echo "Creating local symlink for Blender executable in ${LOCAL_BIN_DIR}..."
    ln -sf "${INSTALL_DIR}/blender" "${LOCAL_BIN_DIR}/blender"
    echo "Local symlink created at ${LOCAL_BIN_DIR}/blender."
else
    echo "WARNING: Blender executable not found at ${INSTALL_DIR}/blender."
fi

# --- Install Dependencies into Blender's Python ---
echo "Installing dependencies into Blender's Python (${BLENDER_PY_EXEC})..."
if [ ! -f "${BLENDER_PY_EXEC}" ]; then
    echo "ERROR: Blender Python executable not found at ${BLENDER_PY_EXEC}!"
    exit 1
fi
if [ ! -f "${UNIRIG_REQS_FILE_IN_SPACE}" ]; then
    echo "ERROR: UniRig requirements file not found at ${UNIRIG_REQS_FILE_IN_SPACE}!"
    exit 1
fi

echo "Upgrading pip for Blender Python..."
"${BLENDER_PY_EXEC}" -m pip install --no-cache-dir --upgrade pip setuptools wheel -vvv

echo "Installing packaging and ninja (recommended for flash-attn)..."
"${BLENDER_PY_EXEC}" -m pip install --no-cache-dir packaging ninja -vvv

echo "Step 1: Installing PyTorch ${TORCH_VERSION} (for CUDA ${TARGET_CUDA_VERSION_SHORT}) and Torchvision ${TORCHVISION_VERSION}..."
"${BLENDER_PY_EXEC}" -m pip install --no-cache-dir \
    torch==${TORCH_VERSION} \
    torchvision==${TORCHVISION_VERSION} \
    --index-url ${TORCH_INDEX_URL} -vvv
echo "PyTorch and Torchvision installation attempted."

echo "Step 2: Installing flash-attn from direct wheel URL..."
# Install flash-attn from a direct wheel URL to ensure compatibility and avoid source build.
# Using --no-deps as we manage other dependencies separately and assume the wheel is self-contained or relies on PyTorch.
"${BLENDER_PY_EXEC}" -m pip install --no-cache-dir \
    --no-deps \
    "${FLASH_ATTN_WHEEL_URL}" -vvv
echo "flash-attn installation attempted from wheel."

echo "Step 3: Installing remaining dependencies from ${UNIRIG_REQS_FILE_IN_SPACE}..."
# Ensure flash-attn is REMOVED from unirig_requirements.txt.
# This will install torch-scatter, torch-cluster, spconv, bpy, etc.
# PyG (torch-scatter, etc.) links in unirig_requirements.txt might need to be updated for torch 2.3 + cu118
# Example: torch-scatter -f https://data.pyg.org/whl/torch-2.3.0+cu118.html (adjust torch version if needed)
"${BLENDER_PY_EXEC}" -m pip install --no-cache-dir \
    -r "${UNIRIG_REQS_FILE_IN_SPACE}" -vvv

echo "Dependency installation for Blender's Python complete."

# --- FIX: Ensure UniRig/src is treated as a package ---
UNIRIG_SRC_DIR="${UNIRIG_REPO_CLONE_DIR}/src"
INIT_PY_PATH="${UNIRIG_SRC_DIR}/__init__.py"
if [ -d "${UNIRIG_SRC_DIR}" ]; then
  if [ ! -f "${INIT_PY_PATH}" ]; then
    echo "Creating missing __init__.py in ${UNIRIG_SRC_DIR}..."
    touch "${INIT_PY_PATH}"
    echo "__init__.py created."
  else
    echo "${INIT_PY_PATH} already exists."
  fi
else
  echo "WARNING: UniRig src directory not found at ${UNIRIG_SRC_DIR}."
fi

# (Optional) VRM Addon installation
VRM_ADDON_REL_PATH="blender/add-on-vrm-v2.20.77_modified.zip"
ABSOLUTE_ADDON_PATH="${UNIRIG_REPO_CLONE_DIR}/${VRM_ADDON_REL_PATH}"
if [ -f "${ABSOLUTE_ADDON_PATH}" ]; then
    echo "Attempting to install optional VRM addon for Blender..."
    (cd "${UNIRIG_REPO_CLONE_DIR}" && \
     "${BLENDER_PY_EXEC}" -c "import bpy, os; print(f'Attempting to install addon from: {os.path.abspath(\"${VRM_ADDON_REL_PATH}\")}'); bpy.ops.preferences.addon_install(overwrite=True, filepath=os.path.abspath('${VRM_ADDON_REL_PATH}')); print('Addon installation script executed. Attempting to enable...'); bpy.ops.preferences.addon_enable(module='io_scene_vrm'); print('VRM Addon enabled successfully.')") \
    || echo "WARNING: VRM addon installation or enabling failed. This is an optional addon. Continuing setup..."
    echo "VRM addon installation/enabling attempt finished."
else
    echo "VRM addon zip not found at ${ABSOLUTE_ADDON_PATH}, skipping addon installation."
fi

# --- Cleanup ---
echo "Cleaning up downloaded Blender tarball..."
rm -f /tmp/${BLENDER_TARBALL}
echo "Cleanup complete."
echo "Blender setup finished successfully. Blender is in ${INSTALL_DIR}"
echo "--- Setup Script End ---"
