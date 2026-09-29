"""I/O helpers shared by the dataset adapters."""
from __future__ import annotations

import json
import os
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Iterator, Tuple


class _NaNToNull:
    """File wrapper turning the non-standard ``NaN`` token into ``null``.

    The challenge files were written by Python's json module, which emits NaN;
    streaming parsers (ijson/yajl) reject it.  A partial token at a chunk
    boundary is carried over to the next read.  Assumes "NaN" never occurs
    inside a string, which holds for these files (hex ids, upper-case codes).
    """

    def __init__(self, f, chunk: int = 1 << 20):
        self.f = f
        self.chunk = chunk
        self.carry = b""

    def read(self, n: int = -1) -> bytes:
        if n == 0:          # ijson probes the stream type with read(0)
            return b""
        size = n if n and n > 0 else self.chunk
        while True:
            data = self.f.read(size)
            if not data:
                out, self.carry = self.carry.replace(b"NaN", b"null"), b""
                return out
            data = self.carry + data
            cut = len(data)
            if data.endswith(b"NaN"):
                pass
            elif data.endswith(b"Na"):
                cut -= 2
            elif data.endswith(b"N"):
                cut -= 1
            out, self.carry = data[:cut], data[cut:]
            if out:
                return out.replace(b"NaN", b"null")


def iter_json_object(path: str) -> Iterator[Tuple[str, object]]:
    """Yield (key, value) of a top-level JSON object, streaming when possible.

    Uses ijson (``pip install ijson``) to keep memory flat on the multi-GB
    travel-time files; falls back to ``json.load`` otherwise.  NaN is mapped to
    None/NaN respectively -- callers must treat both as missing.
    """
    try:
        import ijson
    except ImportError:
        ijson = None
    if ijson is None:
        print(f"[realworld] ijson not installed; loading {os.path.basename(path)} fully into memory",
              file=sys.stderr)
        with open(path) as f:
            yield from json.load(f).items()
        return
    with open(path, "rb") as f:
        yield from ijson.kvitems(_NaNToNull(f), "", use_float=True)


def load_json(path: str):
    with open(path) as f:
        return json.load(f)


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and v != v)


def s3_public_sync(bucket: str, prefix: str, dest: str, region: str = "us-west-2") -> int:
    """Download every object under ``s3://bucket/prefix`` without credentials.

    Equivalent to ``aws s3 sync --no-sign-request s3://bucket/prefix dest``;
    files that already exist with the right size are skipped.  Returns the
    number of files downloaded.
    """
    host = f"https://{bucket}.s3.{region}.amazonaws.com"
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    token, n = None, 0
    while True:
        q = {"list-type": "2", "prefix": prefix}
        if token:
            q["continuation-token"] = token
        with urllib.request.urlopen(f"{host}/?{urllib.parse.urlencode(q)}") as r:
            root = ET.fromstring(r.read())
        for c in root.iter(f"{ns}Contents"):
            key = c.find(f"{ns}Key").text
            size = int(c.find(f"{ns}Size").text)
            if key.endswith("/"):
                continue
            out = os.path.join(dest, key[len(prefix):].lstrip("/"))
            if os.path.exists(out) and os.path.getsize(out) == size:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            print(f"  {key} ({size / 1e6:.1f} MB)")
            tmp = out + ".part"
            urllib.request.urlretrieve(f"{host}/{urllib.parse.quote(key)}", tmp)
            os.replace(tmp, out)
            n += 1
        if root.findtext(f"{ns}IsTruncated") == "true":
            token = root.findtext(f"{ns}NextContinuationToken")
        else:
            return n
