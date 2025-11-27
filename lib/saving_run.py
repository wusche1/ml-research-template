import wandb
import os
import dotenv
from pathlib import Path

dotenv.load_dotenv()

def save_wandb_data(config: dict) -> None:
    """Ask user if they want to keep the wandb run. If not, delete it."""
    if wandb.run is None:
        return
    path = wandb.run.path
    wandb.run.finish()
    api = wandb.Api()
    metrics_dataframe = api.run(path).history()
    metrics_dataframe.to_csv(Path(config["output_dir"]) / "metrics.csv")

    return
    


def handle_huggingface_upload(model, config: dict) -> None:
    """Upload model to HuggingFace Hub."""
    saving = config.get("saving", {}).get("hf", 'not_specified')
        
    if not saving:
        return
    wandb_project = os.environ["WANDB_PROJECT"].replace(" ", "-")
    parent_folder = config["output_dir"].split("/")[-3]
    safe_run_name = config['run_name'].replace("/", "_")
    repo_name = f"{wandb_project}_{parent_folder}_{safe_run_name}"

    if saving == 'not_specified':
        if not input(f"\nSave model to HuggingFace ({repo_name})? (y/n): ").lower().strip() == 'y':
            return

   
    model.push_to_hub(repo_name)
    print(f"Model saved to https://huggingface.co/{repo_name}")

