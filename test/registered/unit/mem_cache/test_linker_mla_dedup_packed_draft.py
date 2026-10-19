"""Every packed draft layer must be broadcast by the linker MLA dedup adapter."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

# Row width per physical layer, so a broadcast of the wrong layer is visible in
# the recorded element size instead of silently reusing a neighbour's geometry.
ROW_BYTES = 10


def _make_pool(layer_mapping):
    return SimpleNamespace(layer_mapping=layer_mapping)


def _make_buffers(num_physical_layers=4, num_components=2):
    """Mirror the adapter's ``buffers[name]`` layout: components of layers."""
    return [
        [
            torch.zeros(4, ROW_BYTES + layer, dtype=torch.uint8)
            for layer in range(num_physical_layers)
        ]
        for _ in range(num_components)
    ]


def _make_broadcaster(layer_mapping, prepared_indices=(0, 1)):
    """Build the adapter's broadcast path without a NCCL group or a GPU."""
    broadcaster = object.__new__(LinkerMLADedupBroadcaster)
    broadcaster.pools = {"full": _make_pool(layer_mapping)}
    broadcaster.buffers = {"full": _make_buffers()}
    broadcaster.kv_staging = torch.zeros(64, dtype=torch.uint8)
    broadcaster.calls = []
    broadcaster._bcast_layer = (
        lambda buf_list, staging, target, elem, layer_id: broadcaster.calls.append(
            (id(buf_list), int(elem), layer_id)
        )
    )
    prepared = {"full": (torch.tensor(prepared_indices), None)}
    return broadcaster, prepared


class TestLinkerMLADedupBroadcastLayers(unittest.TestCase):
    """``layer_mapping`` values are ``int | tuple[int, ...]``.

    ``_with_packed_draft_mapping`` (linker_pool_assembler.py) stores a tuple
    when MTP/EAGLE draft layers are packed into the target pool, and the sibling
    ``get_prepared_layer_range_meta`` already expands those tuples. The dedup
    adapter indexed the per-layer buffer list with the raw mapping value, so a
    packed draft pool either raised ``TypeError`` (tuple index) or, for an
    unpacked pool, was broadcast one layer at a time as intended.
    """

    def test_int_mapping_broadcasts_one_layer(self):
        broadcaster, prepared = _make_broadcaster({0: 2})

        broadcaster.broadcast_loaded_layer(0, prepared)

        components = broadcaster.buffers["full"]
        self.assertEqual(
            broadcaster.calls,
            [(id(component), ROW_BYTES + 2, 2) for component in components],
        )

    def test_packed_draft_tuple_broadcasts_every_physical_layer(self):
        broadcaster, prepared = _make_broadcaster({0: (0, 3)})

        broadcaster.broadcast_loaded_layer(0, prepared)

        components = broadcaster.buffers["full"]
        self.assertEqual(
            broadcaster.calls,
            [
                (id(component), ROW_BYTES + layer, layer)
                for component in components
                for layer in (0, 3)
            ],
        )

    def test_unmapped_layer_broadcasts_nothing(self):
        broadcaster, prepared = _make_broadcaster({0: (0, 3)})

        broadcaster.broadcast_loaded_layer(1, prepared)

        self.assertEqual(broadcaster.calls, [])

    def test_pool_without_prepared_indices_broadcasts_nothing(self):
        broadcaster, _ = _make_broadcaster({0: (0, 3)})

        broadcaster.broadcast_loaded_layer(0, {})

        self.assertEqual(broadcaster.calls, [])


if __name__ == "__main__":
    unittest.main()
