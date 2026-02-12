import subprocess

import mlflow


def get_git_commit():
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def init_mlflow(cfg):
    mlflow.set_tracking_uri(cfg.logging.tracking_uri)
    mlflow.set_experiment(cfg.experiment_name)


def start_run(cfg):
    run = mlflow.start_run(run_name=cfg.run_name)

    if cfg.logging.log.git_commit:
        git_commit = get_git_commit()
        if git_commit:
            mlflow.set_tag("git_commit", git_commit)

    if cfg.logging.log.params:
        mlflow.log_params(
            {
                "seed": cfg.seed,
                "device": cfg.device,
                "model_type": cfg.models.type,
                "model_name": cfg.models.name,
            }
        )

    return run


def log_metrics(metrics, step=None):
    mlflow.log_metrics(metrics, step=step)


def log_params(params):
    mlflow.log_params(params)


def log_artifacts(artifacts_dir):
    mlflow.log_artifacts(artifacts_dir)


def end_run():
    mlflow.end_run()
