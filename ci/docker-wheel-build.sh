#!/usr/bin/env bash

set -e -x

export SUPPORTED_PYTHONS=(cp27-cp27m cp35-cp35m cp36-cp36m)

for PYVER in ${SUPPORTED_PYTHONS[@]}; do
    echo ${PYVER}
    PYBIN=/opt/python/${PYVER}/bin
    "${PYBIN}/pip" install -r /io/ci/requirements_dev.txt
    "${PYBIN}/pip" wheel /io/ --no-deps -w wheelhouse/
done

for whl in wheelhouse/*.whl; do
    auditwheel repair $whl -w /io/wheelhouse/
done

cd $HOME
for PYVER in ${SUPPORTED_PYTHONS[@]}; do
    echo ${PYVER}
    PYBIN=/opt/python/${PYVER}/bin
    ${PYBIN}/pip install randomstate --no-index -f /io/wheelhouse
    ${PYBIN}/pytest --pyargs randomstate
done
