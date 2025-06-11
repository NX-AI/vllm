from dataclasses import dataclass
from typing import Dict, List

import torch

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.attention.backends.utils import PAD_SLOT_ID


@dataclass
class xLSTMCacheParams:
    rnn_state: torch.Tensor = torch.Tensor()
    rnn_normalizer_state: torch.Tensor = torch.Tensor()
    rnn_stabilizer_state: torch.Tensor = torch.Tensor()
    state_indices_tensor: torch.Tensor = torch.Tensor()

    def at_layer_idx(self, layer_idx):
        return xLSTMCacheParams(
            self.rnn_state[layer_idx],
            self.rnn_normalizer_state[layer_idx],
            self.rnn_stabilizer_state[layer_idx],
            self.state_indices_tensor,
        )


class xLSTMCacheManager:
    def __init__(
        self, dtype, num_xlstm_layers, max_batch_size, rnn_cell_shape, rnn_normalizer_shape, rnn_stabilizer_shape
    ):
        print("XLSTM CACHE INIT", num_xlstm_layers, max_batch_size, rnn_cell_shape, dtype)
        rnn_state = torch.empty(size=(num_xlstm_layers, max_batch_size) + rnn_cell_shape, dtype=dtype, device="cuda")
        rnn_normalizer_state = torch.empty(
            size=(num_xlstm_layers, max_batch_size) + rnn_normalizer_shape, dtype=dtype, device="cuda"
        )
        rnn_stabilizer_state = torch.empty(
            size=(num_xlstm_layers, max_batch_size) + rnn_stabilizer_shape, dtype=dtype, device="cuda"
        )
        print("RNN STAB STATE: ", rnn_stabilizer_state.shape, (num_xlstm_layers, max_batch_size), rnn_stabilizer_shape)

        self.xlstm_cache = (rnn_state, rnn_normalizer_state, rnn_stabilizer_state)

        # Maps between the request id and a dict that maps between the seq_id
        # and its index inside the self.xlstm_cache
        self.xlstm_cache_indices_mapping: Dict[str, Dict[int, int]] = {}
        self.free_cache_indices = list(range(max_batch_size))

    def current_run_tensors(self, input_ids: torch.Tensor, attn_metadata: AttentionMetadata, **kwargs):
        """
        Return the tensors for the current run's rnn state.
        """
        if "seqlen_agnostic_capture_inputs" not in kwargs:
            # We get here only on Prefill/Eager mode runs
            request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
            finished_requests_ids = kwargs["finished_requests_ids"]

            self._release_finished_requests(finished_requests_ids)
            state_indices = self._prepare_current_run_xlstm_cache(request_ids_to_seq_ids, finished_requests_ids)

            state_indices_tensor = torch.as_tensor(state_indices, dtype=torch.int32, device="cuda")
            xlstm_cache_tensors = self.xlstm_cache

        else:
            # CUDA graph capturing runs
            (xlstm_cache_tensors, state_indices_tensor) = kwargs["seqlen_agnostic_capture_inputs"]

        return (xlstm_cache_tensors, state_indices_tensor)

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        """
        Copy the relevant state_indices into the CUDA graph input buffer
        """
        assert all(key in kwargs for key in ["request_ids_to_seq_ids", "finished_requests_ids"])
        finished_requests_ids = kwargs["finished_requests_ids"]
        request_ids_to_seq_ids = kwargs["request_ids_to_seq_ids"]
        assert "seqlen_agnostic_capture_inputs" in input_buffers
        _, input_state_indices_buffer = input_buffers["seqlen_agnostic_capture_inputs"]

        self._release_finished_requests(finished_requests_ids)
        state_indices = self._prepare_current_run_xlstm_cache(request_ids_to_seq_ids, finished_requests_ids)
        cuda_graph_pad_len = input_state_indices_buffer.shape[0] - len(state_indices)
        state_indices.extend([PAD_SLOT_ID] * cuda_graph_pad_len)

        input_state_indices_buffer.copy_(torch.as_tensor(state_indices, dtype=torch.int32, device="cuda"))

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        """
        Provide the CUDA graph capture runs with a buffer in adjusted size.
        The buffer is used to maintain the xLSTM Cache during the CUDA graph
        replay runs.
        """
        state_indices_tensor = torch.as_tensor([PAD_SLOT_ID] * batch_size, dtype=torch.int32, device="cuda")
        return (self.xlstm_cache, state_indices_tensor)

    def _copy_xlstm_cache(self, from_index: int, to_index: int):
        assert len(self.xlstm_cache) > 0
        for cache_t in self.xlstm_cache:
            cache_t[:, to_index].copy_(cache_t[:, from_index], non_blocking=True)

    def _assign_seq_id_to_cache_index(self, cur_rid: str, seq_id: int, finished_requests_ids) -> int:
        """
        Assign (req_id,seq_id) pair to a `destination_index` index, if
        already occupied, move the occupying index to a free index.
        """
        if cur_rid in finished_requests_ids:
            # set as pad, do not allocate destination index
            return PAD_SLOT_ID
        elif cur_rid not in self.xlstm_cache_indices_mapping:
            destination_index = self.free_cache_indices.pop()
            self.xlstm_cache_indices_mapping[cur_rid] = {seq_id: destination_index}
            return destination_index
        elif seq_id not in (seq_ids2indices := self.xlstm_cache_indices_mapping[cur_rid]):
            # parallel sampling , where n > 1, assume prefill have
            # already happened, so we copy the
            # existing cache into the siblings seq_ids caches
            index_exists = next(iter(seq_ids2indices.values()))
            # case of decoding n>1, copy prefill cache to decoding indices
            destination_index = self.free_cache_indices.pop()
            self._copy_xlstm_cache(from_index=index_exists, to_index=destination_index)
            self.xlstm_cache_indices_mapping[cur_rid][seq_id] = destination_index
            return destination_index
        else:
            # already exists
            return self.xlstm_cache_indices_mapping[cur_rid][seq_id]

    def _prepare_current_run_xlstm_cache(
        self, request_ids_to_seq_ids: Dict[str, list[int]], finished_requests_ids: List[str]
    ) -> List[int]:
        return [
            self._assign_seq_id_to_cache_index(req_id, seq_id, finished_requests_ids)
            for req_id, seq_ids in request_ids_to_seq_ids.items()
            for seq_id in seq_ids
        ]

    def _release_finished_requests(self, finished_seq_groups_req_ids: List[str]):
        for req_id in finished_seq_groups_req_ids:
            if req_id in self.xlstm_cache_indices_mapping:
                for seq_id in self.xlstm_cache_indices_mapping[req_id]:
                    self.free_cache_indices.append(self.xlstm_cache_indices_mapping[req_id][seq_id])
                self.xlstm_cache_indices_mapping.pop(req_id)
