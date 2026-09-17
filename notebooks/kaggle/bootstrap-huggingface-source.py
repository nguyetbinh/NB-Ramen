"""Apply the bundled acquisition patch as a reproducible, local-only Git commit."""

import os
from pathlib import Path
import subprocess
import sys

BASE_REVISION = "26a7cd7c847b6630841dbae58067e5bb124f2f9d"


def bootstrap(repo, patch):
    repo, patch = Path(repo), Path(patch).resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True)
    if head != BASE_REVISION or dirty:
        raise RuntimeError("Apply Hugging Face support only to the clean pinned base revision")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=repo, check=True)
    subprocess.run(["git", "apply", "--index", str(patch)], cwd=repo, check=True)
    commit_env = dict(os.environ,
        GIT_AUTHOR_NAME="NB-Ramen", GIT_AUTHOR_EMAIL="nb-ramen@localhost",
        GIT_COMMITTER_NAME="NB-Ramen", GIT_COMMITTER_EMAIL="nb-ramen@localhost",
        GIT_AUTHOR_DATE="2026-09-14T00:00:00+00:00", GIT_COMMITTER_DATE="2026-09-14T00:00:00+00:00",
    )
    subprocess.run(["git", "-c", "commit.gpgsign=false", "commit", "-m",
                    "feat(runtime): verify Hugging Face CIFAR-100-C acquisition"],
                   cwd=repo, env=commit_env, check=True)
    print(subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip())


if __name__ == "__main__":
    bootstrap(*sys.argv[1:])
