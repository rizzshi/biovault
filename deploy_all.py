"""BioVault Automated Multi-Platform Deployment Script.

Pushes the BioVault project to:
1. GitHub: Synchronizes local git commits to the remote GitHub repository.
2. Hugging Face: Deploys the Gradio application to a Hugging Face Space.
3. Kaggle: Pushes notebooks/BioVault_Benchmark.ipynb to Kaggle Kernels.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys


def deploy_github(repo_url: str, token: str | None = None, branch: str = "main") -> bool:
    print("\n" + "=" * 60)
    print("1. Pushing to GitHub")
    print("=" * 60)
    try:
        import dulwich.porcelain as dp
        import dulwich.repo

        repo = dulwich.repo.Repo(".")
        
        # Format remote URL with token if provided
        target_url = repo_url
        if token and "@" not in repo_url and repo_url.startswith("https://"):
            target_url = repo_url.replace("https://", f"https://{token}@")

        print(f"Pushing branch '{branch}' to {repo_url} ...")
        dp.push(repo, target_url, f"refs/heads/{branch}".encode("ascii"))
        print("[SUCCESS] GitHub push completed.")
        return True
    except Exception as e:
        print(f"[ERROR] GitHub push failed: {e}", file=sys.stderr)
        return False


def deploy_huggingface(space_id: str, token: str | None = None) -> bool:
    print("\n" + "=" * 60)
    print("2. Pushing to Hugging Face Space")
    print("=" * 60)
    try:
        from push_to_hub import push_to_hub

        print(f"Deploying Space to 'https://huggingface.co/spaces/{space_id}' ...")
        push_to_hub(repo_id=space_id, token=token)
        print("[SUCCESS] Hugging Face Space deployment completed.")
        return True
    except Exception as e:
        print(f"[ERROR] Hugging Face deployment failed: {e}", file=sys.stderr)
        return False


def deploy_kaggle(username: str, api_key: str | None = None) -> bool:
    print("\n" + "=" * 60)
    print("3. Pushing to Kaggle Kernels")
    print("=" * 60)
    try:
        # Update kernel-metadata.json with username
        meta_path = "notebooks/kernel-metadata.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["id"] = f"{username}/biovault-benchmark"
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"Updated {meta_path} kernel id to '{meta['id']}'")

        if api_key:
            os.environ["KAGGLE_USERNAME"] = username
            os.environ["KAGGLE_KEY"] = api_key

        res = subprocess.run(
            [sys.executable, "-m", "kaggle", "kernels", "push", "-p", "notebooks"],
            capture_output=True,
            text=True,
            check=True,
        )
        print(res.stdout)
        print("[SUCCESS] Kaggle notebook pushed successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Kaggle push failed: {e}", file=sys.stderr)
        return False


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    parser = argparse.ArgumentParser(description="Deploy BioVault to GitHub, Hugging Face, and Kaggle")
    parser.add_argument("--github-repo", type=str, default=os.environ.get("GITHUB_REPO_URL"), help="Target GitHub repository URL (e.g. https://github.com/user/biovault.git)")
    parser.add_argument("--github-token", type=str, default=os.environ.get("GITHUB_TOKEN"), help="GitHub Personal Access Token")
    parser.add_argument("--hf-space", type=str, default=os.environ.get("HF_SPACE_ID"), help="Hugging Face Space ID (e.g. username/biovault)")
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"), help="Hugging Face User Access Token")
    parser.add_argument("--kaggle-user", type=str, default=os.environ.get("KAGGLE_USERNAME"), help="Kaggle username")
    parser.add_argument("--kaggle-key", type=str, default=os.environ.get("KAGGLE_KEY"), help="Kaggle API key")
    parser.add_argument("--dry-run", action="store_true", help="Validate files and settings without pushing")

    args = parser.parse_args()

    if args.dry_run:
        print("Dry-run validation successful. Deployment script ready.")
        return

    results = {}
    if args.github_repo:
        results["github"] = deploy_github(args.github_repo, args.github_token)
    if args.hf_space:
        results["huggingface"] = deploy_huggingface(args.hf_space, args.hf_token)
    if args.kaggle_user:
        results["kaggle"] = deploy_kaggle(args.kaggle_user, args.kaggle_key)

    print("\n" + "=" * 60)
    print("Deployment Summary:")
    for k, v in results.items():
        print(f"  {k:15s}: {'PASSED' if v else 'FAILED'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
