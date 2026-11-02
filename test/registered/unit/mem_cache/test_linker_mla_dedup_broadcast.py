"""Unit tests for the packed-draft layer mapping in LinkerMLADedupBroadcaster."""

import unittest
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)


def _broadcaster(layer_mapping):
    # __new__ keeps the test free of a real pool group and NCCL group: only the
    # two attributes read by broadcast_loaded_layer are populated.
    broadcaster = LinkerMLADedupBroadcaster.__new__(LinkerMLADedupBroadcaster)
    broadcaster.pools = {"kv": SimpleNamespace(layer_mapping=layer_mapping)}
    broadcaster.buffers = {
        "kv": [
            [
                torch.empty((4, 128), dtype=torch.uint8),
                torch.empty((4, 256), dtype=torch.uint8),
                torch.empty((4, 512), dtype=torch.uint8),
            ]
        ]
    }
    broadcaster.kv_staging = torch.empty(4096, dtype=torch.uint8)
    calls = []
    broadcaster._bcast_layer = (
        lambda buffers, staging, indices, elem, layer_id: calls.append((elem, layer_id))
    )
    return broadcaster, calls


class TestLinkerMLADedupBroadcast(unittest.TestCase):
    def test_int_layer_mapping_broadcasts_one_physical_layer(self):
        broadcaster, calls = _broadcaster({0: 1})
        prepared = {"kv": (torch.tensor([1, 2]), None)}

        broadcaster.broadcast_loaded_layer(0, prepared)

        self.assertEqual(calls, [(256, 1)])

    def test_packed_draft_tuple_broadcasts_every_physical_layer(self):
        # _with_packed_draft_mapping maps one transfer layer to
        # (target layer, draft layer) when MTP/EAGLE draft KV is packed.
        broadcaster, calls = _broadcaster({0: (1, 2)})
        prepared = {"kv": (torch.tensor([1, 2]), None)}

        broadcaster.broadcast_loaded_layer(0, prepared)

        self.assertEqual(calls, [(256, 1), (512, 2)])

    def test_unmapped_layer_broadcasts_nothing(self):
        broadcaster, calls = _broadcaster({1: 0})
        prepared = {"kv": (torch.tensor([1, 2]), None)}

        broadcaster.broadcast_loaded_layer(0, prepared)

        self.assertEqual(calls, [])

    def test_pool_missing_from_prepared_is_skipped(self):
        broadcaster, calls = _broadcaster({0: (0, 1)})

        broadcaster.broadcast_loaded_layer(0, {})

        self.assertEqual(calls, [])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
