#!/bin/sh

# This script removes components in the default GitHub Actions runner that are not needed if one only needs to build a container image
# Inspiration: https://dev.to/mathio/squeezing-disk-space-from-github-actions-runners-an-engineers-guide-3pjg
# TODO: awfully slow, but it's a price we need to pay unfortunately.

sudo apt-get -y remove --purge \
    '^apache2.*' \
    '^aspnetcore-.*' \
    azure-cli \
    '^byobu.*' \
    '^clang-.*' \
    '^dotnet-.*' \
    firefox \
    '^fonts-.*' \
    fwupd \
    '^gfortran-.*' \
    '^google-.*' \
    '^gradle.*' \
    '^java.*' \
    '^kotlin.*' \
    kubectl \
    '^libclang-.*' \
    libgl1-mesa-dri \
    '^libgirepository-.*' \
    '^libgtk-.*' \
    '^libllvm-.*' \
    '^libx265-.*' \
    '^llvm-.*' \
    man-db \
    '^mecab.*' \
    mediainfo \
    '^mercurial.*' \
    microsoft-edge-stable \
    '^mongodb-.*' \
    '^mono-.*' \
    '^mssql-.*' \
    '^mysql-.*' \
    '^nginx-.*' \
    '^php.*' \
    '^podman.*' \
    '^powershell.*' \
    '^postgres.*' \
    '^ruby.*' \
    '^r-base.*' \
    skopeo \
    tcl \
    tk \
    '^tex-.*' \
    '^vim.*'

sudo apt-get -y remove --purge snapd
sudo rm -rf \
  /snap \
  /usr/lib/snapd \
  /var/snap \
  /var/lib/snapd \
  "${HOME}/snap"

sudo apt autoremove -y && sudo apt clean -y

#
# Now just nuke some directories in case they remain after using apt to clean things up.
#

# Remove Java (JDKs)
sudo rm -rf /usr/lib/jvm &
# Remove .NET SDKs
sudo rm -rf /usr/share/dotnet &
# Remove Swift toolchain
sudo rm -rf /usr/share/swift &
# Remove Haskell (GHC)
sudo rm -rf /usr/local/.ghcup &
# Remove Julia
sudo rm -rf /usr/local/julia* &
# Remove Android SDKs
sudo rm -rf /usr/local/lib/android &
# Remove Chromium (optional if not using for browser tests)
sudo rm -rf /usr/local/share/chromium &
# Remove Microsoft/Edge and Google Chrome builds
sudo rm -rf /opt/microsoft /opt/google &
# Remove Azure CLI
sudo rm -rf /opt/az &
# Remove PowerShell
sudo rm -rf /usr/local/share/powershell &
# Remove CodeQL and other toolcaches
sudo rm -rf /opt/hostedtoolcache &

sudo rm -rf \
  "${HOME}/.rustup" \
  "${HOME}/.cargo" \
  "${HOME}/.dotnet" \
  /opt/mssql-tools \
  /usr/local/bin/aliyun \
  /usr/local/bin/azcopy \
  /usr/local/bin/bicep \
  /usr/local/bin/cmake-gui \
  /usr/local/bin/cpack \
  /usr/local/bin/helm \
  /usr/local/bin/hub \
  /usr/local/bin/kubectl \
  /usr/local/bin/minikube \
  /usr/local/bin/miniconda \
  /usr/local/bin/node \
  /usr/local/lib/node_modules \
  /usr/local/bin/packer \
  /usr/local/bin/pulumi* \
  /usr/local/bin/sam \
  /usr/local/bin/stack \
  /usr/local/bin/oc \
  /usr/local/aws-sam-cli \
  /usr/local/tcltk \
  /usr/local/lib/heroku \
  /usr/local/lib/kotlin* \
  /usr/share/apache-* \
  /usr/share/man/* \
  /usr/share/sbt \
  /usr/local/bin/terraform &
wait
