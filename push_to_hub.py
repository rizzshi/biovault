"""BioVault Hugging Face Hub Deployment Script.

Uploads artifacts to a Hugging Face Space repository:
- checkpoints/global_model.pt
- results/tradeoff_plot.png
- app.py
along with runtime dependencies and telemetry samples.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List


def push_to_hub(
    repo_id: str,
    token: str | None = None,
    create_pr: bool = False,
    dry_run: bool = False,
) -> None:
    """Upload BioVault demo assets to a Hugging Face Space repository.

    Args:
        repo_id: Destination Hugging Face Space repository ID (e.g. 'username/biovault').
        token: Hugging Face user access token (defaults to HF_TOKEN environment variable).
        create_pr: Whether to upload as a Pull Request instead of direct commit.
        dry_run: Validate files and target paths without actually uploading.
    """
    token = token or os.environ.get("HF_TOKEN")

    files_to_upload: List[str] = [
        "app.py",
        "checkpoints/global_model.pt",
        "results/tradeoff_plot.png",
        "requirements.txt",
        "README.md",
        "contracts.py",
        "model.py",
        "data.py",
    ]

    print("=" * 80)
    print("BioVault Hugging Face Space Deployment")
    print("=" * 80)
    print(f"Target Repository ID: {repo_id}")
    print(f"Dry Run Mode:         {dry_run}")
    print("\n[1] Verifying deployment files:")

    missing = []
    for f in files_to_upload:
        if os.path.exists(f):
            sz_kb = os.path.getsize(f) / 1024
            print(f"  [OK] {f:30s} ({sz_kb:8.1f} KB)")
        else:
            print(f"  [MISSING] {f:26s}")
            missing.append(f)

    if missing:
        print(f"\nError: Missing required files for deployment: {missing}", file=sys.stderr)
        sys.exit(1)

    if dry_run:
        print("\n[2] Dry-run completed successfully. All target files verified.")
        return

    if not token:
        print("\nError: Hugging Face token not provided.", file=sys.stderr)
        print("Please set the HF_TOKEN environment variable or pass --token <your_token>.", file=sys.stderr)
        sys.exit(1)

    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        print(f"\n[2] Connecting to Hugging Face Hub (User: {api.whoami()['name']})...")

        # Create Space if not already existing
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            exist_ok=True,
        )

        print("\n[3] Uploading Space assets...")
        for filepath in files_to_upload:
            print(f"  Uploading {filepath} -> {repo_id}:{filepath} ...")
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=filepath,
                repo_id=repo_id,
                repo_type="space",
                create_pr=create_pr,
            )

        # Also upload samples directory
        if os.path.exists("samples"):
            print("  Uploading samples/ directory...")
            api.upload_folder(
                folder_path="samples",
                path_in_repo="samples",
                repo_id=repo_id,
                repo_type="space",
                create_pr=create_pr,
            )

        space_url = f"https://huggingface.co/spaces/{repo_id}"
        print("\n" + "=" * 80)
        print("DEPLOYMENT SUCCESSFUL!")
        print(f"Live Space URL: {space_url}")
        print("=" * 80)

    except Exception as exc:
        print(f"\nDeployment failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload BioVault Demo to Hugging Face Spaces")
    parser.add_argument("--repo-id", type=str, default=os.environ.get("HF_REPO_ID", "biovault-iomt-demo"), help="Target Hugging Face Space repo ID (e.g. username/biovault)")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (or set HF_TOKEN)")
    parser.add_argument("--create-pr", action="store_true", help="Submit as Pull Request instead of direct commit")
    parser.add_argument("--dry-run", action="store_true", help="Verify files without performing network upload")

    args = parser.parse_args()
    push_to_hub(repo_id=args.repo_id, token=args.token, create_pr=args.create_pr, dry_run=args.dry_run)
