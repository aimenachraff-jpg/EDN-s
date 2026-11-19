"""Upload master2-a1-code-v2.zip as a new version of the master2-a1-code dataset.

The v1 dataset on Kaggle shipped with a buggy trainer.py (NameError: 'lr' in
_setup_loss when GAN is enabled) and a realesrgan.py that expected a net_g dict
but got an RRDBNet instance. This script uploads the fixed codebase as version 2.
"""
import json
import os
import sys
import time
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
STAGE = PROJECT_ROOT / "data" / "kaggle_stage_code"
ZIP = STAGE / "master2-a1-code-v2.zip"
OWNER = "aymencharafedn"
SLUG = "master2-a1-code"

tok_path = Path.home() / ".kaggle" / "kaggle.json"
if not os.environ.get("KAGGLE_API_TOKEN") and tok_path.exists():
    os.environ["KAGGLE_API_TOKEN"] = json.loads(tok_path.read_text())["key"]

sys.path.insert(0, str(PROJECT_ROOT))
from kaggle import KaggleApi
from kaggle.api.kaggle_api_extended import ApiBlobType, ResumableUploadContext

api = KaggleApi()
api.authenticate()


def upload_one(path, ctx):
    print(f"  uploading {path.name} ...", flush=True)
    tok = api._upload_blob(path, quiet=False, blob_type=ApiBlobType.DATASET, upload_context=ctx)
    print(f"  token: {tok}", flush=True)
    return tok


def main():
    STAGE.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        print("ERROR: master2-a1-code-v2.zip not found in", STAGE, flush=True)
        sys.exit(1)

    # validate
    with zipfile.ZipFile(ZIP) as z:
        bad = z.testzip()
    if bad:
        print("ERROR: corrupt zip entry:", bad, flush=True)
        sys.exit(1)

    with ResumableUploadContext() as ctx:
        tok = upload_one(ZIP, ctx)

    print("=== creating dataset version ===", flush=True)
    from kagglesdk.datasets.types.dataset_api_service import (
        ApiCreateDatasetVersionRequest,
        ApiCreateDatasetVersionRequestBody,
        ApiDatasetNewFile,
    )
    body = ApiCreateDatasetVersionRequestBody()
    body.version_notes = ("v2: fixed trainer.py _setup_loss (lr NameError when GAN enabled) "
                          "and realesrgan.py net_g constructor (expects config dict, not RRDBNet instance).")
    nf = ApiDatasetNewFile()
    nf.token = tok
    body.files = [nf]
    req = ApiCreateDatasetVersionRequest()
    req.owner_slug = OWNER
    req.dataset_slug = SLUG
    req.body = body
    with api.build_kaggle_client() as kaggle:
        msg = kaggle.datasets.dataset_api_client.create_dataset_version
        resp = api.with_retry(msg)(req)
    print("VERSION RESPONSE:", resp, flush=True)


if __name__ == "__main__":
    main()