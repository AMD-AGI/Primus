#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
HLO IR execution skeleton analyzer.

This tool parses an XLA HLO module text dump, starting from ENTRY, and prints a
hierarchical "execution skeleton" showing while-loops, conditionals, and
selected operation types (communication ops, computation ops, or both).

It strips away non-essential ops and keeps only the control-flow structure and
the relevant ops. All op_name metadata paths are printed in full.

-------------------------------------------------------------------------------
USAGE:
    IRLens_analyze_hlo_ir.py <hlo_ir_dump.txt> [options]

OPTIONS:
    --op {all,communication,computation}
        Select which types of operations to display.
        all (default): show both communication and computation ops
        communication: show only communication ops (all-gather-start, reduce-scatter-start...)
        computation: show only computation ops (fusion kernels, custom-call GEMMs, etc.)

    --name
        Display result variable names and while loop names.

    --topology
        Display communication topology information such as replica_groups or
        source_target_pairs.

    --fusion-stats
        Show detailed fusion subtypes in the statistics summary (skeleton always
        shows the detailed subtype regardless).

DESCRIPTION:
    The output is nested and indented to match the HLO control-flow structure:
        ENTRY %main:
          while i in range(8):
            all-gather-start | bf16[8192] | jit(train_step)/... | file:line
            fusion:mlp_fwd | bf16[...] | ...

    Each displayed operation is printed as:
        op | [name] | payload | [topology] | op_name_path | source_location

    Where:
        - op:                 one of communication ops, fusion subtypes, custom-call subtypes, etc.
        - name (optional):    result variable name in HLO (if --name)
        - payload:            extracted tensor types and shapes
        - topology (optional):replica_groups/source_target_pairs (if --topology)
        - op_name_path:       full metadata op_name="..." path
        - source_location:    "file:line" from the metadata, if present

EXAMPLES:
    # Show all ops in the skeleton (default)
    IRLens_analyze_hlo_ir.py hlo_ir_dump.txt

    # Show only communication ops (EXTREMELY USEFUL!!!)
    IRLens_analyze_hlo_ir.py hlo_ir_dump.txt --op communication

    # Show only computation ops
    IRLens_analyze_hlo_ir.py hlo_ir_dump.txt --op computation

    # Show name and topology fields
    IRLens_analyze_hlo_ir.py hlo_ir_dump.txt --name --topology

    # Break down fusion ops in statistics
    IRLens_analyze_hlo_ir.py hlo_ir_dump.txt --fusion-stats

-------------------------------------------------------------------------------
Output formats:

When --name is OFF and --topology is OFF:
    op | payload | op_name | source_file:source_line

When --name is ON and --topology is OFF:
    op | name | payload | op_name | source_file:source_line

When --name is OFF and --topology is ON:
    op | payload | topology | op_name | source_file:source_line

When --name is ON and --topology is ON:
    op | name | payload | topology | op_name | source_file:source_line
-------------------------------------------------------------------------------
"""

import argparse
import re
import sys
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Set

# --------- Regex patterns for parsing HLO IR ----------

COMP_HEADER_RE = re.compile(r"\s*(ENTRY\s+)?%([^\s(]+)\s*\(.*?\)\s*->.*{")

WHILE_RE = re.compile(r"while\([^)]*\).*condition=%([^\s,}]+),\s*body=%([^\s,}]+)")

COND_TF_RE = re.compile(r"true_computation=%([^\s,}]+),\s*false_computation=%([^\s,}]+)")

BRANCH_COMP_RE = re.compile(r"branch_computations=\{([^}]+)\}")


# --------------------- Utility functions ---------------------


def strip_comment(line: str) -> str:
    """Remove // comments but leave everything else intact."""
    idx = line.find("//")
    return line if idx == -1 else line[:idx]


class LoopInfo:
    """Loop iteration parameters inferred from backend_config and control flow analysis."""

    def __init__(
        self,
        n: Optional[int] = None,
        init: Optional[int] = None,
        step: Optional[int] = None,
        tuple_index: Optional[int] = None,
    ) -> None:
        self.n = n
        self.init = init
        self.step = step
        self.tuple_index = tuple_index


class HLOSimplifier:
    """
    Parse HLO IR and generate execution skeleton with selected operation types.

    Key features:
      * Parse computations from HLO text dump
      * Identify and traverse from ENTRY computation
      * Follow control flow (while loops, conditionals, call operations)
      * Extract communication ops, computation ops, or both based on --op mode
      * Produce hierarchical, indented skeleton output
      * Always print full op_name metadata paths if available
    """

    def __init__(
        self,
        text: str,
        show_name: bool = False,
        show_topology: bool = False,
        op_type: str = "communication",
        show_fusion_details: bool = False,
    ) -> None:
        self.text = text
        self.show_name = show_name
        self.show_topology = show_topology
        self.op_type = op_type
        self.show_fusion_details = show_fusion_details

        # Parsed HLO structure: computation name -> list of lines
        self.computations: Dict[str, List[str]] = {}
        self.entry_candidates: List[str] = []
        self.entry_name: Optional[str] = None

        # Statistics counters
        self.total_communication_ops: int = 0
        self.communication_op_counts: Dict[str, int] = defaultdict(int)

        self.total_computation_ops: int = 0
        self.computation_op_counts: Dict[str, int] = defaultdict(int)

        # Track op_name paths for each displayed operation
        self.all_op_paths: List[str] = []

    # --------------------- HLO text parsing ---------------------

    def parse(self) -> None:
        """Parse HLO text into named computations and identify ENTRY computation."""
        current_comp: Optional[str] = None

        for line in self.text.splitlines():
            m = COMP_HEADER_RE.match(line)
            if m:
                is_entry = m.group(1) is not None
                comp_name = m.group(2)
                current_comp = comp_name
                self.computations[current_comp] = []
                if is_entry:
                    self.entry_candidates.append(comp_name)
                continue

            if current_comp is not None:
                self.computations[current_comp].append(line)

        if not self.entry_candidates:
            raise RuntimeError("No ENTRY computation found")

        # Prefer ENTRY containing "main" if there are multiple candidates
        for name in self.entry_candidates:
            if "main" in name:
                self.entry_name = name
                break

        if self.entry_name is None:
            self.entry_name = self.entry_candidates[0]

    # --------------------- Operation extraction methods ---------------------

    @staticmethod
    def get_op_name(line: str) -> Optional[str]:
        """Extract HLO op name from RHS (e.g., 'all-gather-start', 'fusion', 'custom-call')."""
        line_nc = strip_comment(line)
        if "=" not in line_nc:
            return None
        _, rhs = line_nc.split("=", 1)
        rhs = rhs.strip()

        # Pattern: "...) op_name(..."
        m = re.search(r"\)\s*([A-Za-z0-9_.-]+)\s*\(", rhs)
        if m:
            return m.group(1)

        idx = rhs.find("(")
        if idx == -1:
            return None
        toks = rhs[:idx].split()
        return toks[-1] if toks else None

    @staticmethod
    def get_result_name(line: str) -> Optional[str]:
        """Extract HLO result variable name from LHS (e.g., '%all-gather-start.4')."""
        m = re.search(r"(%[^\s=]+)\s*=", strip_comment(line))
        return m.group(1) if m else None

    @staticmethod
    def get_op_path(line: str) -> Optional[str]:
        """Extract op_name metadata field (hierarchical operation path)."""
        m = re.search(r'op_name="([^"]+)"', line)
        return m.group(1) if m else None

    @staticmethod
    def get_sched_name(line: str) -> Optional[str]:
        """Extract scheduling_name metadata field."""
        m = re.search(r'scheduling_name="([^"]+)"', line)
        return m.group(1) if m else None

    @staticmethod
    def get_source_location(line: str) -> Optional[str]:
        """
        Extract source location from metadata as "source_file:source_line".

        Example: source_file="/workspace/.../layers/quantizations.py" source_line=222
                 Returns: "/workspace/.../layers/quantizations.py:222"
        """
        m_file = re.search(r'source_file="([^"]+)"', line)
        m_line = re.search(r"source_line=(\d+)", line)
        if m_file and m_line:
            return f"{m_file.group(1)}:{m_line.group(1)}"
        if m_file:
            return f"{m_file.group(1)}:?"
        if m_line:
            return f"?:{m_line.group(1)}"
        return None

    def get_computation_op(self, line: str, op_name: str) -> str:
        """
        Extract detailed computation op subtype for display in execution skeleton.

        For custom-call ops:
          Extracts call target from backend_config (e.g., rocblas:gemm_ex, cublas:gemm)

        For fusion ops:
          Always extracts scheduling_name or fusion kind for detailed display
          (--fusion-stats flag only controls statistics aggregation, not display)

        Digit suffixes (e.g., .1, .42) are automatically stripped from all sub-ops.
        """
        if op_name == "custom-call":
            # Extract from call_target_name in backend_config
            m = re.search(r'call_target_name="([^"]+)"', line)
            if m:
                target = m.group(1)
                # Strip digit suffix (e.g., "gemm.1" -> "gemm")
                target = re.sub(r"\.\d+$", "", target)

                # Normalize common library call patterns:
                # __cublas$gemm -> cublas:gemm
                # __cudnn$convForward -> cudnn:convForward
                # rocblas_gemm_ex -> rocblas:gemm_ex
                if target.startswith("__"):
                    target = target.lstrip("_")
                    target = target.replace("$", ":")
                elif "rocblas" in target.lower():
                    parts = target.split("_", 1)
                    if len(parts) == 2:
                        target = f"{parts[0]}:{parts[1]}"
                return f"custom-call:{target}"

            # Fallback: extract from custom_call_target parameter
            m = re.search(r'custom-call[^(]*\([^)]*\),\s*custom_call_target="([^"]+)"', line)
            if m:
                target = re.sub(r"\.\d+$", "", m.group(1))
                return f"custom-call:{target}"

            return "custom-call"

        elif op_name == "fusion":
            # Always extract detailed subtype for display (--fusion-stats controls stats only)

            # Try scheduling_name first (most specific)
            sched = self.get_sched_name(line)
            if sched:
                sched = re.sub(r"\.\d+$", "", sched)
                return f"fusion:{sched}"

            # Try fusion kind from backend_config JSON
            m = re.search(r'"kind":"([^"]+)"', line)
            if m:
                kind = m.group(1)
                # Normalize: kLoop -> loop, kInput -> input
                if kind.startswith("k"):
                    kind = kind[1:].lower()
                return f"fusion:{kind}"

            # Try fusion kind from non-JSON backend_config
            m = re.search(r"kind=k([A-Za-z]+)", line)
            if m:
                return f"fusion:{m.group(1).lower()}"

            return "fusion"

        return op_name

    def get_stats_op(self, detailed_op: str) -> str:
        """
        Map detailed op name to statistics category.

        Collapses fusion:* sub-ops to just "fusion" when --fusion-stats flag is off,
        allowing high-level aggregation in statistics output.
        """
        if not self.show_fusion_details and detailed_op.startswith("fusion:"):
            return "fusion"
        return detailed_op

    def get_communication_op(self, line: str) -> Optional[str]:
        """
        Identify and return canonical communication op.

        Returns the base communication op (e.g., 'all-gather-start', 'all-reduce-start')
        by matching LHS pattern and stripping instance suffixes.

        Skips *-done ops which are async completion markers, not initiations.

        Example: '%all-gather-start.4' -> 'all-gather-start'
        """
        op_name = self.get_op_name(line) or ""

        # Skip *-done ops
        if op_name.endswith("done") or op_name.endswith("-done") or op_name == "async-done":
            return None

        lhs_name = self.get_result_name(line)  # e.g. "%all-gather-start.4"
        if not lhs_name:
            return None

        base = lhs_name.lstrip("%").split(".", 1)[0]  # "all-gather-start"
        if not base.endswith("-start"):
            return None

        return base

    def is_computation_op(self, line: str) -> bool:
        """
        Check if operation has kernel execution time (excludes zero-cost metadata ops).

        Includes: fusion, custom-call, copy, sort, etc.
        Excludes: bitcast, get-tuple-element, tuple, parameter, constant, partition-id

        Also filters out communication ops, control flow, and async-done markers.
        """
        line_nc = strip_comment(line).strip()
        if not line_nc or "=" not in line_nc:
            return False

        # Skip control flow
        if " while(" in line_nc or " conditional(" in line_nc:
            return False

        # Skip communication ops
        if self.get_communication_op(line):
            return False

        # Skip *-done ops
        op_name = self.get_op_name(line) or ""
        if op_name.endswith("done") or op_name.endswith("-done") or op_name == "async-done":
            return False

        # Skip ROOT
        if line_nc.strip().startswith("ROOT"):
            return False

        # Must have an op name
        if not op_name:
            return False

        # Filter out zero-cost metadata/plumbing ops (no kernel launch)
        zero_cost_ops = {
            "bitcast",  # Pure metadata reshape
            "get-tuple-element",  # Pointer dereference
            "tuple",  # Pointer bundling
            "parameter",  # Input placeholder
            "constant",  # Embedded constant data
            "partition-id",  # Device ID query
        }

        if op_name in zero_cost_ops:
            return False

        return True

    def get_payload_summary(self, line: str) -> Optional[str]:
        """
        Generate compact summary of tensor types and shapes from operation result.

        Examples:
          (f32[8192], f32[1024])         -> "f32[8192], f32[1024]"
          ((bf16[...], bf16[...]), ...)  -> "2xbf16[...], 2xbf16[...]"
        """
        line_nc = strip_comment(line)
        if "=" not in line_nc:
            return None
        _, rhs = line_nc.split("=", 1)
        rhs = rhs.strip()

        op_name = self.get_op_name(line_nc)
        if not op_name:
            return None
        idx = rhs.find(op_name)
        if idx == -1:
            return None

        payload = rhs[:idx].strip()
        type_re = re.compile(r"([A-Za-z0-9_]+\[[^\]]*\](?:\{[^}]*\})?)")
        specs = type_re.findall(payload)
        if not specs:
            return None

        counts: "OrderedDict[str, int]" = OrderedDict()
        for spec in specs:
            counts[spec] = counts.get(spec, 0) + 1

        # Use comma-space delimiter
        return ", ".join(f"{count}x{spec}" if count > 1 else spec for spec, count in counts.items())

    # --------- Local topology extraction + async-call fallback ---------

    def _get_topology_info_local(self, line: str) -> Optional[str]:
        """
        Extract topology only from the given line (no cross-computation fallback).
        Used internally by get_topology_info and by async-call resolution.
        """
        s = strip_comment(line)

        # 1) source_target_pairs={{...}} -> return verbatim
        m_st = re.search(r"source_target_pairs=({{.*}})", s)
        if m_st:
            return m_st.group(1)

        # 2) replica_groups={{...}} -> parse into [N,G]<=[R] format
        m_rg = re.search(r"replica_groups=({{.*}})", s)
        if m_rg:
            inner_full = m_rg.group(1)
            # Strip outer braces
            inner = inner_full[2:-2]
            group_strs = [g.strip() for g in inner.split("},{") if g.strip()]
            if not group_strs:
                return None

            num_groups = len(group_strs)
            g0_tokens = [x for x in group_strs[0].split(",") if x.strip()]
            group_size = len(g0_tokens)
            total = group_size * num_groups if group_size and num_groups else None
            if total is None:
                return None

            return f"[{num_groups},{group_size}]<=[{total}]"

        # 3) Compressed replica_groups notation (e.g., [2,8]<=[16] or [8,2]<=[2,8]T(1,0))
        idx = s.find("replica_groups=")
        if idx != -1:
            start = idx + len("replica_groups=")
            i = start
            while i < len(s):
                ch = s[i]
                if ch == ",":
                    # Check if comma starts next field pattern ", <word>="
                    j = i + 1
                    while j < len(s) and s[j].isspace():
                        j += 1
                    k = j
                    while k < len(s) and (s[k].isalnum() or s[k] == "_"):
                        k += 1
                    if k > j and k < len(s) and s[k] == "=":
                        break
                i += 1
            token = s[start:i].strip()
            if token.startswith("[") and "{" not in token:
                return token

        return None

    def get_topology_info(self, line: str) -> Optional[str]:
        """
        Extract and format communication topology information.

        First tries to parse topology from the current line (replica_groups or
        source_target_pairs). If none is found AND the line is an async-start
        with a 'calls=%foo' callee, it will inspect the callee computation
        (e.g. %async_computation.8) to find the actual collective with
        replica_groups and reuse that topology.

        Formats:
          - "[N,G]<=[R]"              Raw replica_groups (N groups, G size, R total replicas)
          - "{{...},{...},...}"       Explicit source_target_pairs
          - "[8,2]<=[2,8]T(1,0)"      Compressed replica_groups notation
        """
        # First, try the current line
        topo = self._get_topology_info_local(line)
        if topo:
            return topo

        # Fallback: async-start / calls=%foo pattern.
        s = strip_comment(line)
        m_calls = re.search(r"calls=%([^\s,}]+)", s)
        if not m_calls:
            return None

        callee = m_calls.group(1).lstrip("%")
        callee_lines = self.computations.get(callee)
        if not callee_lines:
            return None

        # Scan the callee computation for the first line with topology info
        for ln in callee_lines:
            topo = self._get_topology_info_local(ln)
            if topo:
                return topo

        return None

    # --------------------- Loop analysis methods ---------------------

    def _extract_constants(self, lines: List[str]) -> Dict[str, int]:
        consts: Dict[str, int] = {}
        reg = re.compile(r"(%[^\s=]+)\s*=\s*.*constant.*literal=(-?\d+)")
        for ln in lines:
            m = reg.search(strip_comment(ln))
            if m:
                consts[m.group(1)] = int(m.group(2))
        return consts

    def _extract_compare(self, lines: List[str]):
        reg = re.compile(r"compare\((%[^\s,]+),\s*(%[^\s)]+)\).*direction=([A-Z]+)")
        for ln in lines:
            m = reg.search(strip_comment(ln))
            if m:
                return m.group(1), m.group(2), m.group(3)
        return None

    def _extract_step(
        self,
        body: List[str],
        consts: Dict[str, int],
        counter: Optional[str],
    ) -> Optional[int]:
        if not counter:
            return None

        reg = re.compile(r"(%[^\s=]+)\s*=\s*.*(add|subtract)\((%[^\s,]+),\s*(%[^\s)]+)\)")

        for ln in body:
            ln0 = strip_comment(ln)
            m = reg.search(ln0)
            if not m:
                continue
            _, op_name, lhs, rhs = m.group(1), m.group(2), m.group(3), m.group(4)
            if lhs == counter and rhs in consts:
                return consts[rhs] if op_name == "add" else -consts[rhs]
            if rhs == counter and lhs in consts:
                return consts[lhs]
        return None

    def _loop_info_from_while_line(self, line: str) -> LoopInfo:
        info = LoopInfo()

        m = re.search(r'"known_trip_count":\{"n":"(-?\d+)"', line)
        if m:
            info.n = int(m.group(1))

        m = re.search(r'"known_init_step":\{"init":"(-?\d+)","step":"(-?\d+)"', line)
        if m:
            info.init = int(m.group(1))
            info.step = int(m.group(2))

        m = re.search(r'"known_induction_variable":\{"tuple_index":"(\d+)"', line)
        if m:
            info.tuple_index = int(m.group(1))

        return info

    def _infer_loop_info(self, cond: str, body: str) -> LoopInfo:
        """
        Heuristically infer loop parameters from condition and body computations.

        Assumes standard pattern: counter starts at 0, compare(counter, limit) with LT.
        """
        info = LoopInfo()

        cond_lines = self.computations.get(cond)
        if not cond_lines:
            return info

        consts = self._extract_constants(cond_lines)
        cmpinfo = self._extract_compare(cond_lines)
        if not cmpinfo:
            return info

        lhs, rhs, direction = cmpinfo  # direction unused but kept for clarity
        init = 0
        limit: Optional[int] = None
        counter: Optional[str] = None

        if rhs in consts:
            limit = consts[rhs]
            counter = lhs

        body_lines = self.computations.get(body) or []
        step = self._extract_step(body_lines, consts, counter) or 1
        if limit is None:
            return info

        info.n = max(0, limit - init)
        info.init = init
        info.step = step
        return info

    def get_loop_info(self, cond: str, body: str, while_line: str) -> LoopInfo:
        """Combine backend_config metadata with heuristic analysis to extract loop parameters."""
        bc = self._loop_info_from_while_line(while_line)
        inf = self._infer_loop_info(cond, body)

        return LoopInfo(
            n=bc.n if bc.n is not None else inf.n,
            init=bc.init if bc.init is not None else inf.init,
            step=bc.step if bc.step is not None else inf.step,
            tuple_index=bc.tuple_index,
        )

    # --------------------- Skeleton generation ---------------------

    def walk_computation(
        self,
        comp: str,
        indent: int,
        mult: int,
        stack: Optional[Set[str]] = None,
        is_entry: bool = False,
        loop_depth: int = 0,
    ) -> None:
        """
        Recursively traverse computation to generate execution skeleton.

        Prints control flow structures (while, conditional, call) and selected operations
        based on --op mode (communication/computation/all).

        Args:
            comp: Computation name to traverse
            indent: Current indentation level
            mult: Loop multiplier for operation count statistics
            stack: Recursion guard to prevent infinite loops
            is_entry: Whether this is the ENTRY computation
            loop_depth: Current loop nesting depth for unique variable naming
        """
        if stack is None:
            stack = set()

        if comp not in self.computations:
            print(" " * indent + f"# missing %{comp}")
            return

        if comp in stack:
            print(" " * indent + f"# recursive %{comp} skipped")
            return

        stack.add(comp)
        print(" " * indent + ("ENTRY " if is_entry else "") + f"%{comp}:")

        bind = indent + 2
        tab = " " * bind

        for raw in self.computations[comp]:
            line = strip_comment(raw).strip()
            if not line:
                continue

            # -------- While loops --------
            if " while(" in line:
                while_name = self.get_result_name(line) or "<while>"
                m = WHILE_RE.search(line)
                if m:
                    cond, body = m.group(1), m.group(2)
                    info = self.get_loop_info(cond, body, line)
                    comment = f"  # {while_name}" if self.show_name and while_name else ""

                    v = chr(ord("i") + loop_depth)
                    if info.n is not None:
                        if self.show_name and info.tuple_index is not None:
                            v = f"{v}_iv{info.tuple_index}"
                        print(tab + f"while {v} in range({info.n}):{comment}")
                        count = info.n
                    else:
                        print(tab + f"while {v} in range(<unknown>):{comment}")
                        count = 1

                    self.walk_computation(body, bind + 2, mult * count, stack, False, loop_depth + 1)
                    continue

            # -------- Conditional branches --------
            if " conditional(" in line:
                m = COND_TF_RE.search(line)
                if m:
                    t, f = m.group(1), m.group(2)
                    print(tab + f"if %{t}:")
                    self.walk_computation(t, bind + 2, mult, stack, False, loop_depth)
                    print(tab + "else:")
                    self.walk_computation(f, bind + 2, mult, stack, False, loop_depth)
                    continue

                m = BRANCH_COMP_RE.search(line)
                if m:
                    branches = [b.strip().lstrip("%") for b in m.group(1).split(",") if b.strip()]
                    for idx, b in enumerate(branches):
                        kw = "if" if idx == 0 else "elif"
                        print(tab + f"{kw} branch{idx} (%{b}):")
                        self.walk_computation(b, bind + 2, mult, stack, False, loop_depth)
                    continue

            # -------- Call operations --------
            # Extract to_apply=%foo or calls=%foo and recursively walk
            if " call(" in line:
                m_to_apply = re.search(r"to_apply=%([^\s,}]+)", line)
                m_calls = re.search(r"calls=%([^\s,}]+)", line)
                callee = None
                if m_to_apply:
                    callee = m_to_apply.group(1).lstrip("%")
                elif m_calls:
                    callee = m_calls.group(1).lstrip("%")

                if callee:
                    self.walk_computation(callee, bind, mult, stack, False, loop_depth)
                    continue

            # -------- Operation extraction based on --op mode --------
            should_show_communication = self.op_type in ("communication", "all")
            should_show_computation = self.op_type in ("computation", "all")

            if should_show_computation:
                # Extract and display computation operations
                if self.is_computation_op(raw):
                    op_name = self.get_op_name(raw) or "unknown"

                    # Get detailed op for display
                    op = self.get_computation_op(raw, op_name)

                    # Get aggregated op for statistics
                    stats_op = self.get_stats_op(op)

                    payload = self.get_payload_summary(raw)
                    rname = self.get_result_name(raw) or ""
                    src_loc = self.get_source_location(raw)
                    src_field = src_loc or "N/A"

                    op_path = self.get_op_path(raw) or ""
                    self.all_op_paths.append(op_path)

                    # Build output fields (detailed op for display)
                    fields: List[str] = [op]

                    # Optional: result variable name
                    if self.show_name:
                        name_field = rname if rname else "N/A"
                        fields.append(name_field)

                    # Payload (tensor types and shapes)
                    payload_field = payload if payload else "N/A"
                    fields.append(payload_field)

                    # Optional: topology (N/A for computation ops)
                    if self.show_topology:
                        fields.append("N/A")

                    # op_name path placeholder + source location
                    fields.append("<<<OP_PATH>>>")
                    fields.append(src_field)

                    print(tab + " | ".join(fields))

                    # Update statistics (using aggregated op)
                    self.total_computation_ops += mult
                    self.computation_op_counts[stats_op] += mult
                    continue

            if should_show_communication:
                # Extract and display communication operations
                op = self.get_communication_op(raw)
                if op:
                    payload = self.get_payload_summary(raw)
                    rname = self.get_result_name(raw) or ""
                    src_loc = self.get_source_location(raw)
                    src_field = src_loc or "N/A"

                    op_path = self.get_op_path(raw) or ""
                    self.all_op_paths.append(op_path)

                    # Build output fields
                    fields: List[str] = [op]

                    # Optional: result variable name
                    if self.show_name:
                        name_field = rname if rname else "N/A"
                        fields.append(name_field)

                    # Payload (tensor types and shapes)
                    payload_field = payload if payload else "N/A"
                    fields.append(payload_field)

                    # Optional: communication topology
                    if self.show_topology:
                        topo = self.get_topology_info(raw)
                        topo_field = topo if topo else "N/A"
                        fields.append(topo_field)

                    # op_name path placeholder + source location
                    fields.append("<<<OP_PATH>>>")
                    fields.append(src_field)

                    print(tab + " | ".join(fields))

                    # Update statistics
                    self.total_communication_ops += mult
                    self.communication_op_counts[op] += mult

        stack.remove(comp)

    # --------------------- Post-processing ---------------------

    def finalize_op_names(self, original_output: List[str]) -> List[str]:
        """
        Replace <<<OP_PATH>>> placeholders with actual op_name paths.

        Always prints the full op_name path if available; uses "N/A" when missing.
        """
        if not self.all_op_paths:
            return [ln.replace("<<<OP_PATH>>>", "N/A") for ln in original_output]

        cleaned: List[str] = []
        idx = 0

        for ln in original_output:
            if "<<<OP_PATH>>>" not in ln:
                cleaned.append(ln)
                continue

            op_path = self.all_op_paths[idx] if idx < len(self.all_op_paths) else ""
            idx += 1

            replacement = op_path if op_path else "N/A"
            cleaned.append(ln.replace("<<<OP_PATH>>>", replacement))

        return cleaned

    # --------------------- Main execution ---------------------

    def run(self) -> List[str]:
        """
        Main execution: parse HLO, generate skeleton, and produce final output.

        Returns:
            List of output lines ready for printing
        """
        self.parse()

        captured: List[str] = []
        orig_stdout = sys.stdout
        sys.stdout = _ListWriter(captured)

        try:
            self.walk_computation(self.entry_name, 0, 1, set(), True, 0)

            if self.op_type in ("communication", "all"):
                print("\n")
                print(f"# Total communication ops: {self.total_communication_ops}")
                num_op_categories = len(self.communication_op_counts)
                print(f"# {num_op_categories} communication op categories:")
                for op in sorted(self.communication_op_counts):
                    print(f"#   {op:<40} {self.communication_op_counts[op]}")

            if self.op_type in ("computation", "all"):
                print("\n")
                print(f"# Total computation ops: {self.total_computation_ops}")
                num_op_categories = len(self.computation_op_counts)
                print(f"# {num_op_categories} computation op categories:")
                for op in sorted(self.computation_op_counts):
                    print(f"#   {op:<40} {self.computation_op_counts[op]}")
        finally:
            sys.stdout = orig_stdout

        return self.finalize_op_names(captured)


class _ListWriter:
    """Capture stdout into a list for post-processing before final output."""

    def __init__(self, out_list: List[str]) -> None:
        self.out_list = out_list
        self._buf = ""

    def write(self, s: str) -> None:
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            self.out_list.append(line)

    def flush(self) -> None:
        if self._buf:
            self.out_list.append(self._buf)
            self._buf = ""


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description=("Generate execution skeleton from HLO IR dump " "with configurable operation filtering.")
    )
    parser.add_argument("filename", help="Path to HLO text dump file")
    parser.add_argument(
        "--op",
        choices=["all", "communication", "computation"],
        default="all",
        help="Operation types to display: all (default), communication, or computation",
    )
    parser.add_argument(
        "--name",
        action="store_true",
        help="Show HLO result variable names and while loop names",
    )
    parser.add_argument(
        "--topology",
        action="store_true",
        help="Show communication topology info (replica_groups, source_target_pairs)",
    )
    parser.add_argument(
        "--fusion-stats",
        action="store_true",
        help=("Break down fusion ops by subtype in statistics " "(skeleton always shows detail)"),
    )
    args = parser.parse_args(argv)

    with open(args.filename, encoding="utf-8") as f:
        txt = f.read()

    simplifier = HLOSimplifier(
        txt,
        show_name=args.name,
        show_topology=args.topology,
        op_type=args.op,
        show_fusion_details=args.fusion_stats,
    )
    lines = simplifier.run()
    for ln in lines:
        print(ln)


if __name__ == "__main__":
    main()
