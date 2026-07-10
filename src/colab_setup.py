import os
import sys
import shutil


def mount_drive():
    from google.colab import drive
    drive.mount('/content/drive')


def clone_or_pull_repo(repo_url, repo_path="/content/ml_final_project"):
    if not os.path.exists(repo_path):
        os.system(f"git clone {repo_url} {repo_path}")
    else:
        cwd = os.getcwd()
        os.chdir(repo_path)
        os.system("git pull")
        os.chdir(cwd)
    return repo_path


def sync_repo_to_drive(repo_path, drive_repo, exclude=("data",)):
    if not os.path.exists(drive_repo):
        shutil.copytree(repo_path, drive_repo, ignore=shutil.ignore_patterns(*exclude))
        return

    for item in os.listdir(repo_path):
        if item in exclude:
            continue
        src = os.path.join(repo_path, item)
        dst = os.path.join(drive_repo, item)
        if os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
        else:
            shutil.copy2(src, dst)


def setup_project(
    repo_url,
    repo_path="/content/ml_final_project",
    drive_repo="/content/drive/MyDrive/ml_final_project",
):
    mount_drive()
    clone_or_pull_repo(repo_url, repo_path)
    sync_repo_to_drive(repo_path, drive_repo)

    src_path = os.path.join(drive_repo, 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    if drive_repo not in sys.path:
        sys.path.append(drive_repo)

    return drive_repo
