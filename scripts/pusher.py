import os, sys, subprocess, pathlib, argparse, json
ROOT = pathlib.Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except: pass

for p in [r"C:\Users\3xs\AppData\Local\Programs\gh", r"C:\Users\3xs\AppData\Local\Programs\Git\cmd", r"C:\Users\3xs\AppData\Local\Python\pythoncore-3.14-64\Scripts", str(pathlib.Path(sys.executable).parent)]:
    if os.path.exists(p) and p not in os.environ.get("PATH", ""):
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")

def run(cmd, cwd):
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cwd, check=True, shell=os.name=='nt')

def github_push(project_path: pathlib.Path):
    project_path = project_path.resolve()
    if not (project_path / ".git").exists():
        run(["git", "init"], cwd=project_path)
        run(["git", "add", "."], cwd=project_path)
        run(["git", "commit", "-m", "initial commit"], cwd=project_path)
    res = subprocess.run(["git", "remote", "get-url", "origin"], cwd=project_path, capture_output=True, text=True)
    slug = project_path.name.replace(" ", "-").lower()
    if res.returncode != 0:
        try:
            run(["gh", "repo", "create", slug, "--source", ".", "--public"], cwd=project_path)
        except Exception as e:
            print(f"gh repo note: {e}")
        subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{os.getenv('GITHUB_USERNAME', 'rizzshi')}/{slug}.git"], cwd=project_path)
    run(["git", "add", "."], cwd=project_path)
    subprocess.run(["git", "commit", "-m", "auto push"], cwd=project_path)
    run(["git", "push", "-u", "origin", "HEAD"], cwd=project_path)
    print(f"Pushed to GitHub: https://github.com/{os.getenv('GITHUB_USERNAME', 'rizzshi')}/{slug}")

def hf_push(project_path: pathlib.Path):
    from huggingface_hub import HfApi
    api = HfApi()
    username = os.getenv("HF_USERNAME") or "v2xvi"
    slug = project_path.name.replace(" ", "-").lower()
    repo_id = f"{username}/{slug}"
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(project_path),
        ignore_patterns=[".venv/**", ".git/**", "__pycache__/**", "scratch/**", "*.pyc"]
    )
    print(f"Pushed to HF: https://huggingface.co/{repo_id}")

def kaggle_push(project_path: pathlib.Path):
    project_path = project_path.resolve()
    slug = project_path.name.replace(" ", "-").lower()
    if len(slug) < 6:
        slug = f"{slug}-dataset"
    meta = project_path / "dataset-metadata.json"
    username = os.getenv("KAGGLE_USERNAME") or "tfisthis"
    if not meta.exists():
        template = {"title": project_path.name, "id": f"{username}/{slug}", "licenses": [{"name": "mit"}]}
        meta.write_text(json.dumps(template, indent=2))
    try:
        run(["kaggle", "datasets", "create", "-p", str(project_path), "--ignore-patterns", ".env*"], cwd=project_path)
    except:
        run(["kaggle", "datasets", "version", "-p", str(project_path), "-m", "auto update", "--ignore-patterns", ".env*"], cwd=project_path)
    print(f"Pushed to Kaggle: https://www.kaggle.com/datasets/{username}/{slug}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--to", nargs="+", choices=["github","kaggle","hf","all"], default=["all"])
    parser.add_argument("--all-projects", action="store_true")
    args = parser.parse_args()
    targets = ["github","kaggle","hf"] if "all" in args.to else args.to
    projects = []
    if args.all_projects:
        projects = [p.resolve() for p in ROOT.iterdir() if p.is_dir() and not p.name.startswith(".") and p.name not in ["scripts",".git"]]
    else:
        projects = [pathlib.Path(args.project).resolve()]
    for proj in projects:
        print(f"\n=== Pushing {proj.name} ===")
        if "github" in targets:
            try: github_push(proj)
            except Exception as e: print(f"GitHub Error: {e}")
        if "hf" in targets:
            try: hf_push(proj)
            except Exception as e: print(f"HF Error: {e}")
        if "kaggle" in targets:
            try: kaggle_push(proj)
            except Exception as e: print(f"Kaggle Error: {e}")
