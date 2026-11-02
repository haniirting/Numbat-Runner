import os
import subprocess
import sys

# Удаляем только эти старые эксперименты
FILES_TO_DELETE = [
    "scripts/expert_inspector.py",
    "scripts/expert_download_test.py",
    "scripts/weight_decoder_inspector.py",
    ".github/workflows/expert-inspector.yml",
    ".github/workflows/expert-download-test.yml",
    ".github/workflows/weight-decoder-inspector.yml",
    ".github/workflows/launch-model-test.ym",
]

# Этот файл удалит себя после выполнения
SELF = "scripts/cleanup_old_files.py"

def run(*args):
    subprocess.run(args, check=True)

def main():
    print("NUMBAT CLEANUP")
    print("=" * 40)

    # Проверяем, что запускаемся в GitHub Actions
    if not os.getenv("GITHUB_ACTIONS"):
        print("ERROR: This script must run inside GitHub Actions.")
        return 1

    existing = []

    for path in FILES_TO_DELETE:
        if os.path.isfile(path):
            existing.append(path)
            print("[FOUND]", path)
        else:
            print("[SKIP]", path)

    print("\nFiles to delete:", len(existing))

    if not existing:
        print("Nothing to delete.")
        return 0

    # Удаляем старые файлы и сам скрипт
    run("git", "rm", "--", *existing, SELF)

    run("git", "config", "user.name", "github-actions[bot]")
    run(
        "git", "config", "user.email",
        "41898282+github-actions[bot]@users.noreply.github.com"
    )

    run("git", "commit", "-m", "Clean up obsolete Numbat files")
    run("git", "push", "origin", "HEAD:main")

    print("\nCLEANUP SUCCESS")
    print("Old files deleted.")
    print("Cleanup script deleted itself.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("CLEANUP FAILED:", repr(error))
        sys.exit(1)
