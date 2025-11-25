import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")


from research_scaffold import execute_experiments
from research_scaffold.argparsing import get_base_argparser, process_base_args
from functions.compute_steering_vectors import compute_steering_vectors
from functions.introspection_eval import introspection_eval

if __name__ == "__main__":
    args = get_base_argparser().parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    
    execute_experiments(
        function_map={
            "compute_steering_vectors": compute_steering_vectors,
            "introspection_eval": introspection_eval,
        },
        config_path=config_path,
        meta_config_path=meta_config_path,
        sweep_config_path=sweep_config_path,
    )
