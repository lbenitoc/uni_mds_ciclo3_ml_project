import subprocess

if __name__ == "__main__":
    run_id = "8d9baf96edf644de931f3bd345bd650a"
    port = "5001"

    subprocess.run(
        ["mlflow", "models", "serve", "-m", f"runs:/{run_id}/model", "-p", port, "--no-conda"],
        check=True
    )
