# PUSH RULES - ALWAYS FOLLOW
When user says push, push to github/kaggle/hf:
- NEVER ask for tokens. Use gh CLI, huggingface-cli, and kaggle.json which are already authenticated. Tokens are in .env as fallback.
- If user is inside a project, use "." as project path. If user says a project name, use ./project-name
- ALWAYS run: python scripts/pusher.py --project <path> --to <target>
- For all projects: python scripts/pusher.py --all-projects --to <target>
