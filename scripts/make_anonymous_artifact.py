# scripts/make_anonymous_artifact.py
import argparse, os, shutil, zipfile, json, re, sys, tempfile, pathlib

EXCLUDE = {
    ".git", ".github", "runs/docker_heart", "runs/smoke_heart", "__pycache__",
}

PLACEHOLDERS = {
    r"https://github.com/[^ \n]+": "https://anonymous.4open.science/r/faircare-fl",
    r"<YOUR-ORG>/<YOUR-REPO>": "ANON-ORG/ANON-REPO",
}

def anonymize_text(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            txt = f.read()
    except Exception:
        return
    for pat, rep in PLACEHOLDERS.items():
        txt = re.sub(pat, rep, txt)
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=False, help="Include a runs/ directory")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        dst = pathlib.Path(tmp) / "faircare-fl"
        shutil.copytree(".", dst, dirs_exist_ok=True)
        # Remove excluded paths
        for root, dirs, files in os.walk(dst, topdown=True):
            dirs[:] = [d for d in dirs if d not in EXCLUDE]
        # Anonymize text files
        for p in dst.rglob("*"):
            if p.is_file() and p.suffix in {".md",".py",".yml",".yaml",".tex",".txt",".cfg",".toml"}:
                anonymize_text(p)
        # Keep only selected runs
        if args.runs_dir:
            runs_dst = dst / "runs"
            os.makedirs(runs_dst, exist_ok=True)
            shutil.copytree(args.runs_dir, runs_dst / "artifact", dirs_exist_ok=True)
        # Zip
        with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as z:
            for p in dst.rglob("*"):
                z.write(p, p.relative_to(dst))
    print(f"Wrote {args.output}")

if __name__ == "__main__":
    main()
