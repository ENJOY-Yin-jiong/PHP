import torch
import numpy as np

def count_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def load_partial_state_dict(model, state_dict_path):
    state_dict = torch.load(state_dict_path)
    model_state_dict = model.state_dict()

    # 创建一个新的state_dict，只包含匹配的键
    new_state_dict = {}
    for k, v in state_dict.items():
        if k in model_state_dict and v.size() == model_state_dict[k].size():
            new_state_dict[k] = v
        else:
            print(f"Skipping {k} due to size mismatch or missing key")
        
    model.load_state_dict(new_state_dict, strict=False)


def load_partial_state_dict_with_fallback(model, current_params_state_dict, trained_params_state_dict_path):
    """
    Loads model parameters with a fallback strategy, giving precedence to trained parameters,
    then updates or supplements with parameters from the current model's state where keys match and sizes align.

    :param model: The model instance to load parameters into.
    :param current_params_state_dict: State dictionary of the current model's parameters.
    :param trained_params_state_dict_path: File path to the state dictionary of the previously trained model's parameters.
    """
    # Load the state dictionary of the trained model parameters
    trained_state_dict = torch.load(trained_params_state_dict_path)
    
    # Initialize the merged state dictionary with the trained model's parameters as the base
    merged_state_dict = trained_state_dict.copy()
    
    # Iterate through the current model's state dictionary
    for k, v in current_params_state_dict.items():
        # If the key exists in the trained model's state dictionary and sizes match, update with current model's parameter
        if k in trained_state_dict and v.size() == trained_state_dict[k].size():
            print(f"Updating {k} with current parameters due to matching key and size.")
            merged_state_dict[k] = v

    # Load the merged state dictionary into the model, setting strict to False to bypass potential unmatched keys
    model.load_state_dict(merged_state_dict, strict=False)

