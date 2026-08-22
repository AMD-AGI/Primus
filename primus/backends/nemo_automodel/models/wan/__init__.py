"""Wan 2.2 T2V hooks for the AutoModel diffusion backend.

    parallelize.py     repaired parallelization strategy: makes selective AC and
                       reshard_after_forward actually take effect (the in-tree
                       Wan strategy silently ignores both)
    data/              dataloaders: synthetic.py (smoke) and cache.py (real)
"""
