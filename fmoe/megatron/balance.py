r"""
Support for monitoring loss in Megatron
"""
import torch
from fmoe.balance import reset_balance_profile
from fmoe.balance import update_balance_profile
from fmoe.utils import get_torch_default_comm
from .distributed import get_moe_group


balance_dict = {}
num_layers = 0


def reset_gate_hook(_num_layers=None):
    from megatron import get_args

    global balance_dict, num_layers
    if _num_layers is not None:
        num_layers = _num_layers
    reset_balance_profile(balance_dict, num_layers, get_args().balance_strategy)


def get_balance_profile():
    global balance_dict
    return balance_dict


def generate_megatron_gate_hook(layer_idx, num_expert_global):
    from megatron import get_args

    balance_strategy = get_args().balance_strategy

    def megatron_gate_hook(gate_top_k_idx, gate_score_top_k, gate_context):
        global balance_dict
        update_balance_profile(
            balance_dict,
            gate_top_k_idx,
            gate_score_top_k,
            gate_context,
            layer_idx,
            num_expert_global,
            balance_strategy,
        )

    return megatron_gate_hook


def add_balance_log(model, writer, iteration):
    from megatron import is_last_rank

    while hasattr(model, 'module'):
        model = model.module

    balance_dict_tensor = torch.vstack(
        [l.mlp.gate.get_loss(clear=True) for l in model.language_model.transformer.layers]
    ).detach()
    world_group = get_torch_default_comm()
    world_size = torch.distributed.get_world_size(group=world_group)
    torch.distributed.all_reduce(balance_dict_tensor, group=world_group)
    balance_dict_tensor /= world_size

    if writer and is_last_rank():
        for idx, metric_name in enumerate(balance_dict):
            for layer_id, val in enumerate(balance_dict_tensor[idx]):
                writer.add_scalar(
                    f"balance-{metric_name}/layer-{layer_id}", val.item(), iteration
                )
            writer.add_scalar(
                f"balance-{metric_name}/all",
                balance_dict_tensor[idx].mean().item(),
                iteration,
            )


def patch_forward_step(forward_step_func):
    r"""
    Patch model's forward_step_func to support balance loss
    """

    from megatron.mpu import is_pipeline_last_stage
    from megatron import get_args

    if not get_args().balance_strategy:
        return forward_step_func

    def forward_step_with_balance_loss(data_iterator, model, input_tensor):
        args = get_args()
        output = forward_step_func(data_iterator, model, input_tensor)

        if not is_pipeline_last_stage() or not args.balance_strategy or args.balance_strategy == 'naive':
            return output
        loss_name = args.balance_strategy + "_loss"

        while hasattr(model, 'module'):
            model = model.module

        loss_list = [l.mlp.gate.get_loss(clear=False).view(1)
                for l in model.language_model.transformer.layers]
        (loss, state_dict), bal_loss = (
            output,
            torch.cat(loss_list).mean() * args.balance_loss_weight
        )

        # avarage across moe group
        moe_group = get_moe_group()
        world_size = torch.distributed.get_world_size(group=moe_group)
        averaged_bal_loss = bal_loss.clone().detach()
        torch.distributed.all_reduce(averaged_bal_loss, group=moe_group)
        averaged_bal_loss /= world_size

        loss += bal_loss
        state_dict[loss_name] = averaged_bal_loss

        return loss, state_dict

    return forward_step_with_balance_loss


def patch_model_provider(model_provider):
    from megatron import get_args

    def fmoefied_model_provider():
        from .layers import fmoefy
        args = get_args()
        if args.expert_hidden_size is None:
            expert_hidden_size = 4 * args.hidden_size // args.top_k
        else:
            expert_hidden_size = args.expert_hidden_size
        return fmoefy(
            model_provider(),
            num_experts=args.num_experts,
            hidden_hidden_size=expert_hidden_size,
            top_k=args.top_k,
            skip_layer_dist = args.skip_layer_dist
        )

    return fmoefied_model_provider
