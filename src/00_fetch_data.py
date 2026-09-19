#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stage 00 - fetch the two external datasets:

1. TELL (Turkish Electronic Living Lexicon), Inkelas, Kuentay, Orgun & Sprouse,
   UC Berkeley.  Phonemic transcriptions of ~30k lexemes elicited from a native
   speaker in citation, accusative, possessive and other forms.

2. OpenSubtitles-2018 Turkish frequency list from hermitdave/FrequencyWords.
   ~2.0m word types with token counts.  Used only for the plural spot-check,
   because TELL elicited the accusative and possessive but not the plural.

"""
import os
import sys
import urllib.request
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")

TELL_URL = "http://linguistics.berkeley.edu/~tell/telldata_all.zip"
FREQ_URL = ("https://raw.githubusercontent.com/hermitdave/FrequencyWords/"
            "master/content/2018/tr/tr_full.txt")

TELL_TABLES = ["ELICIT.db.txt", "ETYMA.db.txt", "MASTER.db.txt"]


def fetch(url, dest):
    print("  downloading %s" % url)
    req = urllib.request.Request(url, headers={"User-Agent": "research-script/1.0"})
    with urllib.request.urlopen(req, timeout=300) as r, open(dest, "wb") as f:
        f.write(r.read())
    print("  -> %s (%.1f MB)" % (dest, os.path.getsize(dest) / 1e6))


def main():
    os.makedirs(DATA, exist_ok=True)

    print("[1/2] TELL")
    tell_dir = os.path.join(DATA, "tell")
    if all(os.path.exists(os.path.join(tell_dir, t)) for t in TELL_TABLES):
        print("  already present, skipping")
    else:
        zpath = os.path.join(DATA, "telldata_all.zip")
        if not os.path.exists(zpath):
            fetch(TELL_URL, zpath)
        os.makedirs(tell_dir, exist_ok=True)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(tell_dir)
        print("  extracted %d tables" % len(os.listdir(tell_dir)))

    print("[2/2] OpenSubtitles-2018 Turkish frequency list")
    fpath = os.path.join(DATA, "tr_full.txt")
    if os.path.exists(fpath):
        print("  already present, skipping")
    else:
        fetch(FREQ_URL, fpath)

    missing = [t for t in TELL_TABLES if not os.path.exists(os.path.join(tell_dir, t))]
    if missing:
        sys.exit("FATAL: missing TELL tables: %s" % missing)
    print("\nAll data present.")


if __name__ == "__main__":
    main()
