#!/usr/bin/env bash
#
# install_apptainer.sh — Install Apptainer on Ubuntu (non-setuid method)
#
# Author      : Vincenzo Brachetta
# License     : MIT
# Source      : Apptainer Admin Guide — Install Ubuntu Packages
#               https://apptainer.org/docs/admin/main/installation.html#install-ubuntu-packages
# Access date : 19 March 2026
#
# Usage:
#   chmod +x install_apptainer.sh
#   sudo ./install_apptainer.sh
#

set -euo pipefail

# --- helpers -----------------------------------------------------------------

info()  { echo "[INFO]  $*"; }
error() { echo "[ERROR] $*" >&2; exit 1; }

# --- privilege check ---------------------------------------------------------

if [[ "$EUID" -ne 0 ]]; then
    error "Please run as root: sudo ./install_apptainer.sh"
fi

# --- OS check ----------------------------------------------------------------

if ! grep -qi "ubuntu" /etc/os-release 2>/dev/null; then
    error "This script targets Ubuntu. Detected OS: $(. /etc/os-release && echo "$NAME")."
fi

info "Starting Apptainer installation (non-setuid, PPA method)..."

# --- prerequisites -----------------------------------------------------------

info "Updating package lists..."
apt-get update -qq

info "Installing prerequisites..."
apt-get install -y -qq software-properties-common

# --- add PPA and install -----------------------------------------------------

info "Adding Apptainer PPA..."
add-apt-repository -y ppa:apptainer/ppa

info "Updating package lists after PPA addition..."
apt-get update -qq

info "Installing Apptainer..."
apt-get install -y -qq apptainer

# --- verify ------------------------------------------------------------------

if command -v apptainer &>/dev/null; then
    APPTAINER_VERSION=$(apptainer --version)
    info "Installation successful: $APPTAINER_VERSION"
else
    error "Installation failed — 'apptainer' command not found."
fi

info "Done. Run 'apptainer --help' to get started."
