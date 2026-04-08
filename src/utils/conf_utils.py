from omegaconf import OmegaConf

def load_config_with_subset(cfg, subset_key, subset_file_path):
    # Dynamically load and merge the subset file
    subset_config = OmegaConf.load(subset_file_path)
    OmegaConf.update(cfg, subset_key, subset_config, force_add=True)
    return cfg



