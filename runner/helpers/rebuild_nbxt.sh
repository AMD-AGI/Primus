#!/bin/bash
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

export PATH_TO_BNXT_TAR_PACKAGE=${PATH_TO_BNXT_TAR_PACKAGE}

if [[ -f "$PATH_TO_BNXT_TAR_PACKAGE" ]]; then
    LOG_INFO "Rebuilding bnxt from $PATH_TO_BNXT_TAR_PACKAGE ..." && \
    tar xzf "${PATH_TO_BNXT_TAR_PACKAGE}" -C /tmp/ && \
    mv /tmp/libbnxt_re-* /tmp/libbnxt && \
    mv /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so /usr/lib/x86_64-linux-gnu/libibverbs/libbnxt_re-rdmav34.so.inbox && \
    cd /tmp/libbnxt/ && sh ./autogen.sh && ./configure && \
    make -C /tmp/libbnxt clean all install && \
    echo '/usr/local/lib' > /etc/ld.so.conf.d/libbnxt_re.conf && \
    ldconfig && \
    cp -f /tmp/libbnxt/bnxt_re.driver /etc/libibverbs.d/ && \
    cd "${PRIMUS_PATH}" && \
    echo "Rebuilding libbnxt done."
else
    echo "Skip bnxt rebuild. PATH_TO_BNXT_TAR_PACKAGE=$PATH_TO_BNXT_TAR_PACKAGE"
fi
