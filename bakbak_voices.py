#!/usr/bin/env python3
"""List the Bakbak voices of your account:  python bakbak_voices.py [--model m1] [--language hi]
Voice IDs belong to one model, so pick the model first. Put the IDs in settings.env as BAKBAK_VOICE_HI / _PA / _EN."""
import argparse
import os

from dotenv import load_dotenv

import bakbak_client as bb
from settings import load_settings

load_settings()
load_dotenv()
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="")
ap.add_argument("--language", default="")
a = ap.parse_args()
key = os.getenv("BAKBAK_API_KEY", "").strip()
if not key:
    raise SystemExit("Set BAKBAK_API_KEY in .env first.")
voices = bb.list_voices(key, os.getenv("BAKBAK_BASE_URL") or bb.BASE_URL)
rows = [v for v in voices if (not a.model or v.get("model") == a.model)
        and (not a.language or str(v.get("language", "")).lower().startswith(a.language.lower()))]
print(f"{len(rows)} of {len(voices)} voices\n{'model':<9}{'language':<10}{'name':<18}id")
for v in sorted(rows, key=lambda v: (v.get("model", ""), v.get("language", ""), v.get("name", ""))):
    print(f"{v.get('model', '?'):<9}{v.get('language', '?'):<10}{(v.get('name') or '?'):<18}{v.get('id')}")
