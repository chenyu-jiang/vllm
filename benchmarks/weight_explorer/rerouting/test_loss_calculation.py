import torch
import itertools
import random
import numpy as np

def calc_min_loss(expert_outputs, routing_weights, selected_experts, reroute_threshold, experts_to_replace):
    final_hidden_states_per_expert = expert_outputs
    from_experts = experts_to_replace

    # first get the ground truth output for all tokens
    ground_truth_output = torch.zeros((expert_outputs[0].shape[0], selected_experts.shape[1], expert_outputs[0].shape[1]), dtype=expert_outputs[0].dtype, device=expert_outputs[0].device)
    for expert_id in range(8):
        expert_mask = (selected_experts == expert_id)
        expert_weight = (routing_weights * expert_mask)
        ground_truth_output += final_hidden_states_per_expert[expert_id].unsqueeze(1).expand(-1, selected_experts.shape[1], -1) * expert_weight.unsqueeze(-1)
    # we enumerate over all possible dropped experts
    from_expert_tensor = torch.tensor(from_experts, dtype=torch.long, device=expert_outputs[0].device)
    to_experts = sorted(list(set(range(8)) - set(from_experts)))
    reroute_mask = (routing_weights[:, 0] < reroute_threshold).unsqueeze(-1).expand_as(selected_experts) & (torch.isin(selected_experts, from_expert_tensor))
    # calculate loss: each token matching the reroute mask
    # has min loss across all to_experts
    to_experts_losses_per_token = []
    # first assume it is routed to nothing, i.e. zeros
    route_to_zero_loss = torch.linalg.norm((ground_truth_output * reroute_mask.unsqueeze(-1)), dim=-1)
    to_experts_losses_per_token.append(route_to_zero_loss)
    for to_expert in to_experts:
        output = final_hidden_states_per_expert[to_expert].unsqueeze(1).expand(-1, selected_experts.shape[1], -1) * routing_weights.unsqueeze(-1)
        to_expert_loss = torch.linalg.norm((ground_truth_output - output) * reroute_mask.unsqueeze(-1), dim=-1)
        to_experts_losses_per_token.append(to_expert_loss)
    to_experts_losses_per_token = torch.stack(to_experts_losses_per_token, dim=0) # [to_expert, batch, 2]
    # find argmin in to_experts dimension
    min_loss_expert = torch.argmin(to_experts_losses_per_token, dim=0) # [batch, 2]
    min_loss = torch.min(to_experts_losses_per_token, dim=0)[0].sum().item()

    min_loss_achieved = min_loss
    optimal_min_loss_expert_map = min_loss_expert * reroute_mask
    value_tensor = torch.tensor([-1] + to_experts, dtype=torch.long, device=optimal_min_loss_expert_map.device)
    map_shape = optimal_min_loss_expert_map.shape
    optimal_selected_expert = torch.index_select(value_tensor, 0, optimal_min_loss_expert_map.view(-1)).view(map_shape)
    import code
    code.interact(local=locals())
    return min_loss_achieved, optimal_selected_expert, torch.min(to_experts_losses_per_token, dim=0)[0]

def reference_calc_min_loss(expert_outputs, routing_weights, selected_experts, reroute_threshold, experts_to_replace):
    min_loss_expert = []
    min_losses = []
    for token_idx in range(routing_weights.shape[0]):
        current_token_min_loss_expert = []
        current_token_min_loss = []
        for expert_top_k_idx in range(routing_weights.shape[1]):
            selected_expert = selected_experts[token_idx, expert_top_k_idx].item()
            if selected_expert not in experts_to_replace or routing_weights[token_idx, 0] >= reroute_threshold:
                current_token_min_loss_expert.append(0)
                current_token_min_loss.append(0)
                continue
            ground_truth_output = expert_outputs[selected_expert][token_idx] * routing_weights[token_idx, expert_top_k_idx]
            losses = []
            loss_if_zero = torch.linalg.norm(ground_truth_output)
            losses.append(loss_if_zero.item())
            for expert_idx in range(8):
                if expert_idx in experts_to_replace:
                    continue
                expert_output = expert_outputs[expert_idx][token_idx] * routing_weights[token_idx, expert_top_k_idx]
                diff_norm = torch.linalg.norm(ground_truth_output - expert_output)
                losses.append(diff_norm.item())
            min_expert_idx = torch.argmin(torch.tensor(losses))
            min_loss = losses[min_expert_idx]
            current_token_min_loss_expert.append(min_expert_idx)
            current_token_min_loss.append(min_loss)
        min_loss_expert.append(current_token_min_loss_expert)
        min_losses.append(current_token_min_loss)
    optimal_selected_expert = []
    value_tensor = [-1] + list(set(range(8)) - set(experts_to_replace))
    for token_idx in range(routing_weights.shape[0]):
        current_token_optimal_selected_expert = []
        for expert_top_k_idx in range(routing_weights.shape[1]):
            current_token_optimal_selected_expert.append(value_tensor[min_loss_expert[token_idx][expert_top_k_idx]])
        optimal_selected_expert.append(current_token_optimal_selected_expert)
    optimal_selected_expert = torch.tensor(optimal_selected_expert, dtype=torch.long, device=routing_weights.device)
    min_loss = torch.tensor(min_losses).sum().item()
    return min_loss, optimal_selected_expert, min_losses


def test_calc_min_loss(k_experts_to_replace: int):
    B = 32
    H = 4096
    ground_truth_experts = []
    for token_idx in range(B):
        # generate 2 ground truth experts without replacement
        expert_1 = random.randint(0, 7)
        expert_2 = random.randint(0, 7)
        while expert_2 == expert_1:
            expert_2 = random.randint(0, 7)
        ground_truth_experts.append([expert_1, expert_2])
    selected_experts = torch.tensor(ground_truth_experts, dtype=torch.long)
    # generate expert weights
    expert_weights = torch.randn(B, 2)
    expert_weights = torch.softmax(expert_weights, dim=-1)
    expert_weights = torch.sort(expert_weights, dim=-1, descending=True)[0] / torch.sum(expert_weights, dim=-1, keepdim=True)
    # generate experts to replace
    experts_to_replace = random.sample(range(8), k_experts_to_replace)
    # generate expert outputs
    expert_outputs = []
    for expert_id in range(8):
        expert_outputs.append(torch.randn(B, H))

    experts_to_maintain = list(set(range(8)) - set(experts_to_replace))
    print("Setting expert {} output to {}".format(experts_to_maintain[0], experts_to_replace[0]))
    expert_outputs[experts_to_maintain[0]] = expert_outputs[experts_to_replace[0]]

    reroute_threshold = 0.9
    min_loss, min_loss_expert_map, min_losses_per_token = calc_min_loss(expert_outputs, expert_weights, selected_experts, reroute_threshold, experts_to_replace)
    reference_min_loss, reference_min_loss_expert_map, reference_min_losses_per_token = reference_calc_min_loss(expert_outputs, expert_weights, selected_experts, reroute_threshold, experts_to_replace)
    reference_min_losses_per_token = torch.tensor(reference_min_losses_per_token)
    try:
        assert torch.allclose(min_loss_expert_map, reference_min_loss_expert_map)
        assert torch.allclose(torch.tensor(min_loss), torch.tensor(reference_min_loss))
        assert torch.allclose(min_losses_per_token, reference_min_losses_per_token)
        import code
        code.interact(local=locals())
    except AssertionError:
        import code
        code.interact(local=locals())

if __name__ == "__main__":
    # for _ in range(50):
    test_calc_min_loss(2)