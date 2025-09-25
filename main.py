import os
import hydra
from hydra.core.hydra_config import HydraConfig


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg):
    output_dir = HydraConfig.get().runtime.output_dir
    if cfg.Method == "DSB":
        from run_dsb import run
        return run(cfg, output_dir)
    elif cfg.Method == "DBDSB":
        from run_dbdsb import run
        return run(cfg, output_dir)
    elif cfg.Method == "RF":
        from run_rf import run
        return run(cfg, output_dir)
    else: 
        raise NotImplementedError

if __name__ == "__main__":
    main()