from research_scaffold import execute_experiments
from research_scaffold.argparsing import get_base_argparser, process_base_args
from functions.train_sft import train_sft

if __name__ == "__main__":
    args = get_base_argparser().parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    
    execute_experiments(
        function_map={"train_sft": train_sft},
        config_path=config_path,
        meta_config_path=meta_config_path,
        sweep_config_path=sweep_config_path,
    )
