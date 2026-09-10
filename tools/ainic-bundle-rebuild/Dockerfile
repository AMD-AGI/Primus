# Rebuild a published training image against a pre-downloaded AINIC bundle.
# Unpack the tarball inside the image and install libionic from the matching
# distro directory. See docs/04-technical-guides/ainic-bundle-versions.md §7.
ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG AINIC_BUNDLE

RUN mkdir /tmp/ainic
COPY ${AINIC_BUNDLE} /tmp/ainic/bundle.tar.gz
RUN set -ex; \
    rm -f /etc/apt/sources.list.d/*amdainic*; \
    cd /tmp/ainic; \
    tar -xf bundle.tar.gz; \
    cd */; \
    tar -xf host_sw_pkg.tar*; \
    cd host_sw_pkg/ionic_driver/deb; \
    tar -xf libionic-debs.tar* 2>/dev/null || tar -xf rdma-core-debs.tar*; \
    REL_NAME="$(. /etc/os-release; echo "$VERSION_CODENAME")"; \
    dpkg -i --force-overwrite "$REL_NAME"/libionic1_*.deb; \
    dpkg -i --force-overwrite "$REL_NAME"/libionic-dev_*.deb; \
    cd /tmp; \
    rm -rf /tmp/ainic

# Structural check: the package is installed consistently and the provider
# symlink the fabric actually loads points at the installed object.
RUN set -eux; \
    dpkg-query -W -f='${Package} ${Version}\n' libionic1 libionic-dev; \
    dpkg -C; \
    ls /usr/lib/x86_64-linux-gnu/libibverbs/libionic-rdmav*.so; \
    readlink -f /usr/lib/x86_64-linux-gnu/libionic.so.1
