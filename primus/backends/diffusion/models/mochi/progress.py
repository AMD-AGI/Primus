###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################
#
# No-op progress bar. The official Mochi VAE decode path updates a tqdm bar;
# training only needs the encoder.


class _ProgressBar:
    def update(self, _count: int = 1):
        return None


def get_new_progress_bar(*args, **kwargs):
    del args, kwargs
    return _ProgressBar()
