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

def get_project_slug(project_path: pathlib.Path, explicit_name: str = None) -> tuple[str, str]:
    if explicit_name:
        slug = explicit_name.replace(" ", "-").lower()
        return slug, explicit_name
    
    meta_path = project_path / "dataset-metadata.json"
    if meta_path.exists():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            if "id" in m and "/" in m["id"]:
                return m["id"].split("/")[1], m.get("title", project_path.name)
        except Exception:
            pass

    citation = project_path / "CITATION.cff"
    if citation.exists():
        for line in citation.read_text(encoding="utf-8").splitlines():
            if line.startswith("title:"):
                raw_title = line.split(":", 1)[1].strip(" '\"")
                slug = raw_title.split(":")[0].strip().replace(" ", "-").lower()
                return slug, raw_title

    readme = project_path / "README.md"
    if readme.exists():
        for line in readme.read_text(encoding="utf-8").splitlines():
            if line.startswith("title:"):
                raw_title = line.split(":", 1)[1].strip(" '\"")
                slug = raw_title.split("—")[0].split(":")[0].strip().replace(" ", "-").lower()
                return slug, raw_title
            if line.startswith("# "):
                raw_title = line.replace("# ", "").strip()
                slug = raw_title.split(":")[0].strip().replace(" ", "-").lower()
                return slug, raw_title

    slug = project_path.name.replace(" ", "-").lower()
    return slug, project_path.name

def github_push(project_path: pathlib.Path, name_override: str = None):
    project_path = project_path.resolve()
    slug, title = get_project_slug(project_path, name_override)
    if not (project_path / ".git").exists():
        run(["git", "init"], cwd=project_path)
        run(["git", "add", "."], cwd=project_path)
        run(["git", "commit", "-m", "initial commit"], cwd=project_path)
    res = subprocess.run(["git", "remote", "get-url", "origin"], cwd=project_path, capture_output=True, text=True)
    if res.returncode != 0:
        try:
            run(["gh", "repo", "create", slug, "--source", ".", "--public"], cwd=project_path)
        except Exception as e:
            print(f"gh repo note: {e}")
        subprocess.run(["git", "remote", "add", "origin", f"https://github.com/{os.getenv('GITHUB_USERNAME', 'rizzshi')}/{slug}.git"], cwd=project_path)
    run(["git", "add", "."], cwd=project_path)
    subprocess.run(["git", "commit", "-m", f"update {slug}"], cwd=project_path)
    subprocess.run(["git", "pull", "--rebase", "origin", "main"], cwd=project_path)
    run(["git", "push", "-u", "origin", "HEAD"], cwd=project_path)
    print(f"Pushed to GitHub: https://github.com/{os.getenv('GITHUB_USERNAME', 'rizzshi')}/{slug}")

def hf_push(project_path: pathlib.Path, name_override: str = None):
    from huggingface_hub import HfApi
    api = HfApi()
    username = os.getenv("HF_USERNAME") or "v2xvi"
    slug, title = get_project_slug(project_path, name_override)
    repo_id = f"{username}/{slug}"
    api.create_repo(repo_id=repo_id, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(project_path),
        ignore_patterns=[".venv/**", ".git/**", "__pycache__/**", "scratch/**", "*.pyc"]
    )
    print(f"Pushed to HF: https://huggingface.co/{repo_id}")

def kaggle_push(project_path: pathlib.Path, name_override: str = None):
    project_path = project_path.resolve()
    slug, title = get_project_slug(project_path, name_override)
    if len(slug) < 6:
        slug = f"{slug}-dataset"
    meta = project_path / "dataset-metadata.json"
    username = os.getenv("KAGGLE_USERNAME") or "tfisthis"
    if not meta.exists():
        template = {"title": title, "id": f"{username}/{slug}", "licenses": [{"name": "mit"}]}
        meta.write_text(json.dumps(template, indent=2))
    cmd_create = ["kaggle", "datasets", "create", "-p", str(project_path), "--ignore-patterns", ".env*"]
    print(f"$ {' '.join(cmd_create)}")
    res = subprocess.run(cmd_create, cwd=project_path, capture_output=True, text=True, shell=os.name=='nt')
    if "already in use" in res.stdout.lower() or res.returncode != 0:
        run(["kaggle", "datasets", "version", "-p", str(project_path), "-m", "auto update", "--ignore-patterns", ".env*"], cwd=project_path)
    else:
        print(res.stdout)
    print(f"Pushed to Kaggle: https://www.kaggle.com/datasets/{username}/{slug}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default=".")
    parser.add_argument("--name", default=None, help="Explicit project name / slug override")
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
        slug, title = get_project_slug(proj, args.name)
        print(f"\n=== Pushing {title} ({slug}) ===")
        if "github" in targets:
            try: github_push(proj, args.name)
            except Exception as e: print(f"GitHub Error: {e}")
        if "hf" in targets:
            try: hf_push(proj, args.name)
            except Exception as e: print(f"HF Error: {e}")
        if "kaggle" in targets:
            try: kaggle_push(proj, args.name)
            except Exception as e: print(f"Kaggle Error: {e}")
