from research_scaffold import execute_experiments
from research_scaffold.argparsing import get_base_argparser, process_base_args
from functions.generate_completions import generate_completions
from functions.load_hf_completions import load_hf_completions
from functions.rate_completions import rate_completions

if __name__ == "__main__":
    args = get_base_argparser().parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    
    execute_experiments(
        function_map={
            "generate_completions": generate_completions,
            "load_hf_completions": load_hf_completions,
            "rate_completions": rate_completions,
        },
        config_path=config_path,
        meta_config_path=meta_config_path,
        sweep_config_path=sweep_config_path,
    )

