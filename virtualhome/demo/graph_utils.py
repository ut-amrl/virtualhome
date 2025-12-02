from collections import defaultdict
from typing import Dict, List, Tuple, Union
import requests
import pandas as pd
import re
import random
from utils_demo import *
import json
import csv

def prepare_pano_character_camera(comm):
    s, msg = comm.add_character_camera(position=[0, 1.8,  0.0], rotation=[30,  0, 0], field_view=60, name="pano_0")
    s, msg = comm.add_character_camera(position=[0, 1.8,  0.0], rotation=[30, 60, 0], field_view=60, name="pano_1")
    s, msg = comm.add_character_camera(position=[0, 1.8,  0.0], rotation=[30, 120, 0], field_view=60, name="pano_2")
    s, msg = comm.add_character_camera(position=[0, 1.8,  0.0], rotation=[30, 180, 0], field_view=60, name="pano_3")
    s, msg = comm.add_character_camera(position=[0, 1.8,  0.0], rotation=[30, 240, 0], field_view=60, name="pano_4")
    s, msg = comm.add_character_camera(position=[0, 1.8,  0.0], rotation=[30, 300, 0], field_view=60, name="pano_5")
    
def prepare_tall_pano_character_camera(comm):
    s, msg = comm.add_character_camera(position=[0, 2.3,  0.0], rotation=[30,  0, 0], field_view=60,  name="tall_pano_0")
    s, msg = comm.add_character_camera(position=[0, 2.3,  0.0], rotation=[30, 60, 0], field_view=60,  name="tall_pano_1")
    s, msg = comm.add_character_camera(position=[0, 2.3,  0.0], rotation=[30, 120, 0], field_view=60, name="tall_pano_2")
    s, msg = comm.add_character_camera(position=[0, 2.3,  0.0], rotation=[30, 180, 0], field_view=60, name="tall_pano_3")
    s, msg = comm.add_character_camera(position=[0, 2.3,  0.0], rotation=[30, 240, 0], field_view=60, name="tall_pano_4")
    s, msg = comm.add_character_camera(position=[0, 2.3,  0.0], rotation=[30, 300, 0], field_view=60, name="tall_pano_5")
    
def extract_nodes_by_ids(nodes, id_set):
    """Return nodes whose 'id' is in the given id_set."""
    return [node for node in nodes if node['id'] in id_set]

def load_prefab_metadata(prefab_path: str = "../resources/PrefabClass.json") -> dict:
    """
    Load prefab metadata from a JSON file.
    
    Args:
        prefab_path (str): Path to the JSON file containing prefab metadata.
        
    Returns:
        dict: Parsed JSON data as a dictionary.
    """
    with open(prefab_path, 'r') as f:
        prefab_data = json.load(f)
        prefab_classes = {}
        classes = []
        for prefab_class in prefab_data["prefabClasses"]:
            cls = prefab_class["className"].replace(" ", "").replace("_", "").lower()
            prefab_classes[cls] = prefab_class["prefabs"]
            classes.append(cls)
    return prefab_classes, classes

def semantic_cls_to_rgb(class_name: str, class_list: list):
    BINS_PER_CHANNEL = 9
    CHANNEL_GAP = 255 // (BINS_PER_CHANNEL - 1)  # 36

    name = class_name.replace(" ", "").replace("_", "").lower()
    try:
        idx = class_list.index(name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' not found.")

    r = ((idx // (BINS_PER_CHANNEL * BINS_PER_CHANNEL)) % BINS_PER_CHANNEL) * CHANNEL_GAP
    g = ((idx // BINS_PER_CHANNEL) % BINS_PER_CHANNEL) * CHANNEL_GAP
    b = (idx % BINS_PER_CHANNEL) * CHANNEL_GAP
    return (r, g, b)

def semantic_cls_to_bgr(class_name: str, class_list: list):
    BINS_PER_CHANNEL = 9
    CHANNEL_GAP = 255 // (BINS_PER_CHANNEL - 1)  # 36

    name = class_name.replace(" ", "").replace("_", "").lower()
    try:
        idx = class_list.index(name)
    except ValueError:
        raise ValueError(f"Class '{class_name}' not found.")

    r = ((idx // (BINS_PER_CHANNEL * BINS_PER_CHANNEL)) % BINS_PER_CHANNEL) * CHANNEL_GAP
    g = ((idx // BINS_PER_CHANNEL) % BINS_PER_CHANNEL) * CHANNEL_GAP
    b = (idx % BINS_PER_CHANNEL) * CHANNEL_GAP
    return (b, g, r)

def semantic_rgb_to_cls(rgb: tuple, class_list: list):
    # Reconstruct index → class name list
    BINS_PER_CHANNEL = 9
    CHANNEL_GAP = 255 // (BINS_PER_CHANNEL - 1)  
    
    r, g, b = rgb
    # TODO need to double check this to ensure nothing is off by 1
    r_bin = r // CHANNEL_GAP
    g_bin = g // CHANNEL_GAP
    b_bin = b // CHANNEL_GAP
    # r_bin = round(r / CHANNEL_GAP)
    # g_bin = round(g / CHANNEL_GAP)
    # b_bin = round(b / CHANNEL_GAP)
    idx = r_bin * 81 + g_bin * 9 + b_bin

    if idx >= len(class_list):
        return "unknown"
    return class_list[idx]

def bgr_to_rgb_imgs(imgs):
    return [img[:, :, ::-1].copy() for img in imgs]