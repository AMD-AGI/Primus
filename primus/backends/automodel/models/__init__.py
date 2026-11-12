###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Per-model support for the AutoModel diffusion backend.

Two invariants hold across this package:

- **No re-exports.** Modules are reached by their dotted path, so adding a model
  does not mean editing anything here. Nothing in this tree should have to be
  touched to add the next one.
- **A model subpackage never imports another's.** Anything two models need lives
  above them -- in ``distributed/``, ``profiling/`` or ``quantization/`` -- not
  in whichever model happened to need it first.
"""
