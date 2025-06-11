"""PyTorch xLSTM Model."""

from typing import Optional, Tuple, List, Set, Iterable

import torch
import torch.utils.checkpoint
from torch import nn

from vllm.attention.backends.abstract import AttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.sampler import SamplerOutput, get_sampler
from vllm.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.interfaces import (
    HasInnerState,
    IsAttentionFree,
    SupportsPP,
)
from vllm.model_executor.models.xlstm_cache import (
    xLSTMCacheManager,
    xLSTMCacheParams,
)
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors
from vllm.utils import LayerBlockType

from .utils import (
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

try:
    from xlstm.xlstm_large.model import (
        RMSNorm,
        mLSTMBlock,
        xLSTMLargeConfig,
    )
except ImportError:
    mLSTMBlock = None
    xLSTMLargeConfig = None

KVCache = Tuple[torch.Tensor, torch.Tensor]


class xLSTMModel(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()

        vllm_config.model_config.hf_config.num_hidden_layers = vllm_config.model_config.hf_config.num_blocks
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.padding_idx = config.pad_token_id
        lora_vocab = (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1)) if lora_config else 0
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.embeddings = VocabParallelEmbedding(
            self.vocab_size,
            config.embedding_dim,
            org_num_embeddings=config.vocab_size,
        )

        # use config explicitly to mitigate unused variable tests
        xlstm_block_config = xLSTMLargeConfig(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            num_blocks=config.num_blocks,
            num_heads=config.num_heads,
            use_bias=config.use_bias,
            add_out_norm=config.add_out_norm,
            norm_eps=config.norm_eps,
            norm_reduction_force_float32=config.norm_reduction_force_float32,
            # mlstm_layer
            qk_dim_factor=config.qk_dim_factor,
            v_dim_factor=config.v_dim_factor,
            # mlstm backend
            chunkwise_kernel=config.chunkwise_kernel,
            sequence_kernel=config.sequence_kernel,
            step_kernel=config.step_kernel,
            mode=config.mode,
            chunk_size=config.chunk_size,
            return_last_states=config.return_last_states,
            autocast_kernel_dtype=config.autocast_kernel_dtype,
            eps=config.eps,
            inference_state_dtype=config.inference_state_dtype,
            # feedforward
            ffn_proj_factor=config.ffn_proj_factor,
            ffn_round_up_to_multiple_of=config.ffn_round_up_to_multiple_of,
            # capping
            gate_soft_cap=config.gate_soft_cap,
            output_logit_soft_cap=config.output_logit_soft_cap,
            weight_mode=config.weight_mode,
        )

        self.start_block, self.end_block, self.blocks = make_layers(
            config.num_blocks,
            lambda prefix: mLSTMBlock(xlstm_block_config),
            prefix=f"{prefix}.layers",
        )

        self.out_norm = RMSNorm(config.embedding_dim, eps=config.norm_eps)

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.embedding_dim
        )

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attn_metadata: AttentionMetadata,
        xlstm_cache_params: xLSTMCacheParams,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        print(
            "INPUT SHAPES",
            input_ids.shape,
            positions.shape,
            inputs_embeds.shape if inputs_embeds is not None else None,
            input_ids,
            # xlstm_cache_params,
        )
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]

        # TODO!
        hidden_states = hidden_states.view(self.max_batch_size, *hidden_states.shape)
        for i in range(self.start_block, self.end_block):
            xlstm_block = self.blocks[i]
            hidden_states, rnn_state = xlstm_block(
                hidden_states,
                state=None
                if attn_metadata.context_lens_tensor is None
                else (
                    xlstm_cache_params.rnn_state[i - self.start_block],
                    xlstm_cache_params.rnn_normalizer_state[i - self.start_block],
                    xlstm_cache_params.rnn_stabilizer_state[i - self.start_block],
                ),
            )
            xlstm_cache_params.rnn_state[i - self.start_block] = rnn_state
        hidden_states = hidden_states.view(hidden_states.shape[0] * hidden_states.shape[1], *hidden_states.shape[2:])
        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.out_norm(hidden_states, residual)

        return hidden_states


class xLSTMForCausalLM(nn.Module, HasInnerState, IsAttentionFree, SupportsPP):
    def __init__(self, *, vllm_config, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        lora_config = vllm_config.lora_config
        scheduler_config = vllm_config.scheduler_config
        assert not cache_config.enable_prefix_caching, "xLSTM does not support prefix caching"

        super().__init__()
        self.config = config
        self.vllm_config = vllm_config
        self.scheduler_config = scheduler_config
        self.model_config = vllm_config.model_config

        self.backbone = xLSTMModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "backbone"))
        self.unpadded_vocab_size = config.vocab_size
        if lora_config:
            self.unpadded_vocab_size += lora_config.lora_extra_vocab_size
        if config.tie_word_embeddings:
            self.lm_head = self.backbone.embeddings
        else:
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                config.embedding_dim,
                org_num_embeddings=config.vocab_size,
                padding_size=DEFAULT_VOCAB_PADDING_SIZE
                # We need bigger padding if using lora for kernel
                # compatibility
                if not lora_config
                else lora_config.lora_vocab_padding_size,
            )

        self.xlstm_cache: Optional[xLSTMCacheManager] = None

        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size,
            config.vocab_size,
            soft_cap=config.output_logit_soft_cap,
        )
        self.sampler = get_sampler()

        self.make_empty_intermediate_tensors = self.backbone.make_empty_intermediate_tensors
        if self.scheduler_config is not None and not self.model_config.enforce_eager:
            if self.scheduler_config.max_num_seqs > vllm_config.compilation_config.max_capture_size:
                self.max_batch_size = vllm_config.compilation_config.max_capture_size
            else:
                self.max_batch_size = vllm_config.pad_for_cudagraph(self.scheduler_config.max_num_seqs)
        else:
            self.max_batch_size = 8192 + 2

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.backbone.get_input_embeddings(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if self.xlstm_cache is None:
            num_xlstm_layers = self.config.num_blocks
            self.xlstm_cache = xLSTMCacheManager(
                self.lm_head.weight.dtype,
                num_xlstm_layers,
                self.max_batch_size,
                *self._get_xlstm_cache_shape(),
            )

        (
            xlstm_cache_tensors,
            state_indices_tensor,
        ) = self.xlstm_cache.current_run_tensors(input_ids, attn_metadata, **kwargs)
        xlstm_cache_params = xLSTMCacheParams(
            xlstm_cache_tensors[0], xlstm_cache_tensors[1], xlstm_cache_tensors[2], state_indices_tensor
        )

        hidden_states = self.backbone(
            input_ids,
            positions,
            attn_metadata,
            xlstm_cache_params,
            intermediate_tensors,
            inputs_embeds,
        )

        return hidden_states

    def copy_inputs_before_cuda_graphs(self, input_buffers, **kwargs):
        return self.xlstm_cache.copy_inputs_before_cuda_graphs(input_buffers, **kwargs)

    def get_seqlen_agnostic_capture_inputs(self, batch_size: int):
        return self.xlstm_cache.get_seqlen_agnostic_capture_inputs(batch_size)

    def _get_xlstm_cache_shape(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        world_size = get_tensor_model_parallel_world_size()
        rnn_state_shape = (
            self.config.num_heads,
            self.config.qk_head_dim,
            self.config.v_head_dim // world_size,
        )
        rnn_normalizer_shape = (
            self.config.num_heads,
            self.config.qk_head_dim,
        )
        rnn_stabilizer_shape = (
            self.config.num_heads,
            1,
        )
        return rnn_state_shape, rnn_normalizer_shape, rnn_stabilizer_shape

    def compute_logits(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata) -> torch.Tensor:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits

    def sample(
        self,
        logits: Optional[torch.Tensor],
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> Set[str]:
        params_dict = dict(self.named_parameters())
        loaded_params: Set[str] = set()
        for name, loaded_weight in weights:
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue
            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params
