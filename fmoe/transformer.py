r"""
Adaption to act as the MLP layer using an MoE MLP layer in transformer.
"""
import torch
import torch.nn as nn
from .gates import NaiveGate
from .layers import FMoE, FMoELinear

class MegatronBaseMLP(torch.nn.Module):
    def __init__(self, hidden_size, activation=torch.nn.GELU()):
        super(MegatronBaseMLP, self).__init__()

        self.htoh4 = torch.nn.Linear(hidden_size, 2 * hidden_size)
        self.activation = activation
        self.h4toh = torch.nn.Linear(2 * hidden_size, hidden_size)

    def forward(self, input):
        x = self.htoh4(input)
        x = self.activation(x)
        x = self.h4toh(x)
        return x


class _Expert(nn.Module):
    r"""
    An expert using 2 FMoELinear modules to speed up the computation of experts
    within one worker.
    """

    def __init__(self, num_expert, d_model, d_hidden, activation, rank=0):
        super().__init__()
        self.htoh4 = FMoELinear(num_expert, d_model, d_hidden, bias=True, rank=rank)
        self.h4toh = FMoELinear(num_expert, d_hidden, d_model, bias=True, rank=rank)
        self.activation = activation

    def forward(self, inp, fwd_expert_count):
        r"""
        First expand input to 4h (the hidden size is variable, but is called h4
        for convenience). Then perform activation. Finally shirink back to h.
        """
        x = self.htoh4(inp, fwd_expert_count)
        x = self.activation(x)
        x = self.h4toh(x, fwd_expert_count)
        return x


class FMoETransformerMLP(FMoE):
    r"""
    A complete MoE MLP module in a Transformer block.
    * `activation` is the activation function to be used in MLP in each expert.
    * `d_hidden` is the dimension of the MLP layer.
    """

    def __init__(
        self,
        num_expert=32,
        d_model=1024,
        d_hidden=4096,
        world_size=1,
        mp_group=None,
        moe_group=None,
        activation=torch.nn.GELU(),
        gate=NaiveGate,
        top_k=2,
        expert_dp_comm="none",
        gate_hook=None,
        mask=None,
        mask_dict=None,
        gate_all_comm=True,
        layer_idx = -1
    ):
        super().__init__(
            num_expert=num_expert,
            d_model=d_model,
            gate=gate,
            top_k=top_k,
            world_size=world_size,
            mp_group=mp_group,
            moe_group=moe_group,
            expert=MegatronBaseMLP,
            gate_hook=gate_hook,
            mask=mask,
            mask_dict=mask_dict,
            gate_all_comm = gate_all_comm,
            layer_idx=layer_idx
        )
        #self.experts = _Expert(
        #    num_expert, d_model, d_hidden, activation, rank=self.mp_rank
        #)
        self.mark_parallel_comm(expert_dp_comm)

    def forward(self, inp: torch.Tensor):
        r"""
        This module wraps up the FMoE module with reshape, residual and layer
        normalization.
        """
        original_shape = inp.shape
        inp = inp.reshape(-1, self.d_model)
        output = super().forward(inp)
        # if torch.distributed.get_rank()==0:
        #     print("model/origin shape:", self.d_model,original_shape)
        return output.reshape(original_shape)
