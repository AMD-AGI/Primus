###############################################################################
# Some parts of this code are copied and modified from
# Sea AI Lab's zero-bubble-pipeline-parallelism project
# (https://github.com/sail-sg/zero-bubble-pipeline-parallelism).
#
# Modification Copyright© 2025 Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
###############################################################################

import collections


class RollbackDataIteratorWrapper:

    def __init__(self, data_iterator):
        self._data_iterator = data_iterator
        self._buffer = collections.deque()
        self._save_to_buffer = False
        self._pop_from_buffer = True

    def save_to_buffer(self):
        self._save_to_buffer = True
        self._pop_from_buffer = False

    def pop_from_buffer(self):
        self._save_to_buffer = False
        self._pop_from_buffer = True

    def clear_buffer(self):
        self._buffer.clear()

    def __iter__(self):
        return self

    def __next__(self):
        if self._pop_from_buffer:
            assert not self._save_to_buffer
            if len(self._buffer) > 0:
                return self._buffer.popleft()
        elem = next(self._data_iterator)
        if self._save_to_buffer:
            assert not self._pop_from_buffer
            self._buffer.append(elem)
        return elem
