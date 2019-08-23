#!/usr/bin/env bash

# Following steps from:
# https://www.tensorflow.org/install/source#common_installation_problems

# Install pre-requisites for tensorflow
pip install -U --user pip six numpy wheel setuptools mock future>=0.17.1
pip install -U --user keras_applications==1.0.6 --no-deps
pip install -U --user keras_preprocessing==1.0.5 --no-deps

# install Bazel (compiles tensorflow)
sudo apt-get install pkg-config zip g++ zlib1g-dev unzip python3
curl -O https://github.com/bazelbuild/bazel/releases/download/0.28.1/bazel-0.28.1-installer-linux-x86_64.sh
chmod +x bazel-0.28.1-installer-linux-x86_64.sh
./bazel-0.28.1-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
echo "deb [arch=amd64] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
sudo apt-get update && sudo apt-get install bazel

