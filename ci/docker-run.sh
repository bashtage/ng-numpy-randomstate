#!/usr/bin/env bash
# Usage:
#
#   ci/docker-run.sh
#
# Must be started from repo root

rm -rf wheelhouse
mkdir wheelhouse

export DOCKER_IMAGE=quay.io/pypa/manylinux1_x86_64
sudo docker pull $DOCKER_IMAGE
sudo docker run --rm -v `pwd`:/io $DOCKER_IMAGE /io/ci/docker-wheel-build.sh
