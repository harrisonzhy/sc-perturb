import argparse
import os
import sys
from pathlib import Path

def main():
    try:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
    except Exception as e:
        print("ERROR: This script requires the 'huggingface_hub' package.\n"
              "Install it with: pip install --upgrade huggingface_hub", file=sys.stderr)
        sys.exit(1)

    p = argparse.ArgumentParser(description="Download a checkpoint (.ckpt) from Hugging Face Hub")
    p.add_argument("--ckpt-repo", default="jkobject/scPRINT",
                   help="Hugging Face repo id, e.g. 'owner/repo'. Default: %(default)s")
    p.add_argument("--ckpt-file", default="small.ckpt",
                   help="File path within the repo (relative). Default: %(default)s")
    p.add_argument("--revision", default=None,
                   help="Optional revision (branch, tag, or commit SHA).")
    p.add_argument("--hf-token", default=None,
                   help="HF token (optional). Falls back to HF_TOKEN env var if not provided.")
    p.add_argument("--output-dir", default=".",
                   help="Directory to place the downloaded file. Default: current directory.")
    p.add_argument("--cache-dir", default=None,
                   help="Optional local cache directory for HF Hub.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing local file if present.")
    args = p.parse_args()

    token = args.hf_token or os.environ.get("HF_TOKEN")
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / Path(args.ckpt_file).name

    if out_path.exists() and not args.overwrite:
        print(f"Already exists: {out_path} (use --overwrite to re-download)")
        print(str(out_path))
        return

    try:
        local_path = hf_hub_download(
            repo_id=args.ckpt_repo,
            filename=args.ckpt_file,
            revision=args.revision,
            token=token,
            cache_dir=args.cache_dir,
            repo_type="model",
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
        )
    except HfHubHTTPError as e:
        print(f"Hub error: {e}", file=sys.stderr)
        sys.exit(2)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        sys.exit(3)
    
    print(f"Downloaded to: {local_path}")
    print(str(local_path))

if __name__ == "__main__":
    main()

