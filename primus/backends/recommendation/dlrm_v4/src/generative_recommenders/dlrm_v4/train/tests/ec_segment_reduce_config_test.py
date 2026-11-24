"""Strict Gin registration and independent EC policy configuration."""

import subprocess
import sys
import textwrap
import unittest


class ECSegmentReduceConfigTest(unittest.TestCase):
    def test_strict_gin_before_lazy_gpu_import_and_independent_rollback(self):
        script = textwrap.dedent(r"""
            import sys
            import types
            import gin
            from generative_recommenders.dlrm_v4.train import ec_bf16_output as ec

            prefix = 'generative_recommenders.dlrm_v4.train.'
            names = [prefix + name for name in
                     ('ec_segment_reduce', 'ec_gpu_plan', 'ec_offsets')]
            assert all(name not in sys.modules for name in names)
            # Registration and strict parsing must work before the GPU setter
            # is imported. A stand-in lets this contract run without a GPU.
            gin.parse_config('enable_segment_reduce.gpu_planner = True\n'
                             'enable_segment_reduce.fixed_size_offsets = True')
            assert all(name not in sys.modules for name in names)
            calls = []
            setter = types.ModuleType(names[0])
            setter.configure_segment_reduce = lambda **kwargs: calls.append(kwargs)
            sys.modules[names[0]] = setter
            for planner, offsets in ((False, False), (True, False),
                                     (False, True), (True, True), (False, False)):
                gin.clear_config()
                gin.parse_config(
                    f'enable_segment_reduce.gpu_planner = {planner}\n'
                    f'enable_segment_reduce.fixed_size_offsets = {offsets}')
                ec.enable_segment_reduce()
                assert calls[-1]['gpu_planner'] is planner
                assert calls[-1]['fixed_size_offsets'] is offsets
                # synthetic re-derivation defaults (s15p1b run 70728)
                assert calls[-1]['split_above'] == 8192
                assert calls[-1]['block_s'] == 16
                assert calls[-1]['num_warps'] == 2
                assert calls[-1]['block_w_cap'] == 512
            gin.clear_config()
            ec.enable_segment_reduce()
            assert calls[-1]['gpu_planner'] is False
            assert calls[-1]['fixed_size_offsets'] is False
            assert all(name not in sys.modules for name in names[1:])
            for name in ('gpu_planner', 'fixed_size_offsets'):
                for value in (0, 1, 'False', None):
                    before = len(calls)
                    try:
                        ec.enable_segment_reduce(**{name: value})
                    except ValueError:
                        pass
                    else:
                        raise AssertionError((name, value, 'accepted'))
                    assert len(calls) == before
        """)
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
