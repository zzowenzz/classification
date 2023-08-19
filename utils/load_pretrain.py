import torch

def load_pretrain(net, path):
    pretrained_state_dict = torch.load(path)

    net_keys = list(net.state_dict().keys())
    pretrained_keys = list(pretrained_state_dict.keys())

    # Check if the keys match in order and presence
    if net_keys == pretrained_keys:
        net.load_state_dict(pretrained_state_dict)
        print("Successfully loaded pretrained weights!")
    else:
        # Find the first mismatched layer for better error reporting
        mismatched_layer = None
        for n_key, p_key in zip(net_keys, pretrained_keys):
            if n_key != p_key:
                mismatched_layer = (n_key, p_key)
                break
        
        if mismatched_layer:
            print(f"Mismatched layers: net has {mismatched_layer[0]} but pretrained has {mismatched_layer[1]}")
        
        raise ValueError("Network architecture does not match pretrained weights!")