# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe
"""CPU tests for ipc_mode.py (ROCr IPC-mode log line). No GPU, no torch."""

import os
import tempfile
import unittest

from generative_recommenders.dlrm_v4.train import ipc_mode

# A /proc/net/unix excerpt as seen on a CV350 node during training in DMA-BUF mode: the
# header, unnamed sockets (7 fields), other abstract names, and the NUL-padded ROCr names.
_HEADER = "Num       RefCount Protocol Flags    Type St Inode Path\n"
_DMABUF = _HEADER + (
    "0000000000000000: 00000002 00000000 00010000 0001 01 4123 @ISCSIADM_ABSTRACT_NAMESPACE\n"
    "0000000000000000: 00000003 00000000 00000000 0001 03 4124\n"
    "0000000000000000: 00000002 00000000 00010000 0001 01 51346 @hsa346" + "@" * 100 + "\n"
    "0000000000000000: 00000002 00000000 00010000 0001 01 51347 @hsa347" + "@" * 100 + "\n"
    "0000000000000000: 00000002 00000000 00010000 0001 01 51348 @hsa3480" + "@" * 99 + "\n"
    "0000000000000000: 00000002 00000000 00010000 0001 01 51349 @hsax12@@@\n"
    "0000000000000000: 00000002 00000000 00010000 0001 01 51350 /run/hsa346.sock\n"
)
_LEGACY = _HEADER + (
    "0000000000000000: 00000002 00000000 00010000 0001 01 4123 @ISCSIADM_ABSTRACT_NAMESPACE\n"
    "0000000000000000: 00000003 00000000 00000000 0001 03 4124\n"
)


class IpcModeTest(unittest.TestCase):
    def _file(self, text: str) -> str:
        fd, path = tempfile.mkstemp()
        with os.fdopen(fd, "w") as f:
            f.write(text)
        self.addCleanup(os.unlink, path)
        return path

    def test_socket_pids_parses_only_rocr_names(self) -> None:
        pids = ipc_mode._socket_pids(_DMABUF.splitlines())
        self.assertEqual(pids, {346, 347, 3480})

    def test_dmabuf_mode(self) -> None:
        info = ipc_mode.detect_rocr_ipc_mode(pid=346, net_unix=self._file(_DMABUF), environ={})
        self.assertEqual(info["mode"], "dmabuf")
        self.assertTrue(info["own_socket"])
        self.assertEqual(info["node_sockets"], 3)
        self.assertIsNone(info["env"])

    def test_pid_prefix_is_not_a_match(self) -> None:
        # pid 348 must not match the socket of pid 3480.
        info = ipc_mode.detect_rocr_ipc_mode(pid=348, net_unix=self._file(_DMABUF), environ={})
        self.assertFalse(info["own_socket"])
        self.assertEqual(info["mode"], "unknown")

    def test_legacy_mode(self) -> None:
        env = {"HSA_ENABLE_IPC_MODE_LEGACY": "1"}
        info = ipc_mode.detect_rocr_ipc_mode(pid=346, net_unix=self._file(_LEGACY), environ=env)
        self.assertEqual(info["mode"], "legacy")
        self.assertFalse(info["own_socket"])
        self.assertEqual(info["node_sockets"], 0)

    def test_mismatch(self) -> None:
        env = {"HSA_ENABLE_IPC_MODE_LEGACY": "1"}
        info = ipc_mode.detect_rocr_ipc_mode(pid=347, net_unix=self._file(_DMABUF), environ=env)
        self.assertEqual(info["mode"], "mismatch")

    def test_env_zero_without_socket_is_unknown(self) -> None:
        env = {"HSA_ENABLE_IPC_MODE_LEGACY": "0"}
        info = ipc_mode.detect_rocr_ipc_mode(pid=1, net_unix=self._file(_LEGACY), environ=env)
        self.assertEqual(info["mode"], "unknown")

    def test_unreadable(self) -> None:
        info = ipc_mode.detect_rocr_ipc_mode(pid=1, net_unix="/nonexistent/net/unix", environ={})
        self.assertEqual(info["mode"], "unreadable")

    def test_log_never_raises_and_logs_local_rank0(self) -> None:
        with self.assertLogs(ipc_mode.logger, level="WARNING") as cm:
            info = ipc_mode.log_rocr_ipc_mode(rank=0, local_rank=0, world_size=8)
        self.assertIsNotNone(info)
        self.assertTrue(any("[ipc] ROCr IPC mode=" in r for r in cm.output))


if __name__ == "__main__":
    unittest.main()
