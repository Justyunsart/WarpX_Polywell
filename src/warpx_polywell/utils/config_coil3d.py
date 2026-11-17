from dataclasses import asdict 
from pathlib import Path 
import json

post_init_keys = [
    'R_coil',
    'dx',
    'eps',
    'm_i',
    'rho',
    'P_ram',
    'P_th',
    'B_ref',
    'd_i',
    'dt',
    'n_floor',
]

def save_cfg_to_json(cfg, filepath: str) -> None:
    """Converts a dataclass instance to JSON and saves it to a file."""
    # Convert dataclass to a standard dictionary
    data_dict = asdict(cfg)
    # these are instantiated using initial arguments, hence not needed for init of PolywellConfig
    for post_init_key in post_init_keys:
        data_dict.pop(post_init_key)
    filename = 'config_used.json'
    filepath = Path(filepath, filename)
    # Save the dictionary as a pretty-printed JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data_dict, f, indent=4)
    print(f"Successfully saved data to {filepath}")

def load_cfg_from_json(filepath: str):
    from inputs.coil3D.coil_3d import SingleCoil3DConfig
    """Loads a JSON file and unpacks it back into the specified dataclass."""
    with open(filepath, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)
        
    # Unpack the dictionary keys as keyword arguments (**kwargs) into the class
    return SingleCoil3DConfig(**data_dict)