"""Download the ListenBrainz sample dump (235 MB) and stream it into compact parquet.

The archive is never unpacked to disk (it is ~2.3 GB uncompressed); each member is
read from the zstd stream and only the needed columns are written to data/lb/.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
import zstandard

from .config import LB, LB_SAMPLE, LB_SAMPLE_SHA256, LB_SAMPLE_URL, RAW

TAG_FILES = {  # member name -> source label
    "recording_tag.parquet": "recording",
    "recording_genre.parquet": "recording",
    "release_group_tag.parquet": "release_group",
    "release_group_genre.parquet": "release_group",
    "artist_tag.parquet": "artist",
    "artist_genre.parquet": "artist",
}


def download() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    if LB_SAMPLE.exists() and _sha256(LB_SAMPLE) == LB_SAMPLE_SHA256:
        print("[extract] sample dump already downloaded")
        return
    print("[extract] downloading ListenBrainz sample dump (235 MB) ...")
    tmp = LB_SAMPLE.with_suffix(".part")
    with requests.get(LB_SAMPLE_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    if _sha256(tmp) != LB_SAMPLE_SHA256:
        tmp.unlink()
        raise SystemExit("checksum mismatch - download corrupted, try again")
    tmp.replace(LB_SAMPLE)


def _sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _recordings(fileobj) -> pd.DataFrame:
    rows = []
    for line in fileobj:  # bytes lines; json.loads accepts utf-8 bytes
        d = json.loads(line)
        rel = d.get("release_data") or {}
        art = d.get("artist_data") or {}
        rows.append((
            d["recording_mbid"], (d.get("recording_data") or {}).get("name"),
            art.get("name"), "|".join(d.get("artist_mbids") or []),
            rel.get("name"), rel.get("year"), rel.get("release_group_mbid"),
            rel.get("caa_id"), rel.get("caa_release_mbid"),
        ))
    return pd.DataFrame(rows, columns=["recording_mbid", "title", "artist", "artist_mbids", "release",
                                       "year", "release_group_mbid", "caa_id", "caa_release_mbid"])


def run() -> None:
    t0 = time.time()
    download()
    LB.mkdir(parents=True, exist_ok=True)
    tags = []
    with open(LB_SAMPLE, "rb") as fh:
        stream = zstandard.ZstdDecompressor(max_window_size=2**31).stream_reader(fh)
        with tarfile.open(fileobj=stream, mode="r|") as tar:
            for m in tar:
                name = m.name.rsplit("/", 1)[-1]
                if name == "recordings_cache.jsonl":
                    df = _recordings(tar.extractfile(m))
                    df["year"] = pd.to_numeric(df.year, errors="coerce").astype("Int16")
                    df["caa_id"] = pd.to_numeric(df.caa_id, errors="coerce").astype("Int64")
                    df.to_parquet(LB / "recordings.parquet", index=False)
                    print(f"[extract] recordings: {len(df):,}")
                elif name in TAG_FILES:
                    t = pq.read_table(io.BytesIO(tar.extractfile(m).read()))
                    t = t.rename_columns(["recording_mbid", "tag", "count"])
                    t = t.append_column("source", pa.array([TAG_FILES[name]] * t.num_rows))
                    tags.append(t)
                elif name == "recording_feedback.parquet":
                    pq.write_table(pq.read_table(io.BytesIO(tar.extractfile(m).read())), LB / "feedback.parquet")
                elif name == "recording_length.parquet":
                    t = pq.read_table(io.BytesIO(tar.extractfile(m).read()), columns=["recording_mbid", "length"])
                    pq.write_table(t, LB / "length.parquet")
                elif name == "top_recording.csv":
                    df = pd.read_csv(io.BytesIO(tar.extractfile(m).read()),
                                     usecols=["recording_mbid", "total_listen_count", "total_user_count"])
                    df.to_parquet(LB / "popularity.parquet", index=False)
                    print(f"[extract] popularity rows: {len(df):,}")
    tag_table = pa.concat_tables(tags)
    pq.write_table(tag_table, LB / "mb_tags.parquet", compression="zstd")
    print(f"[extract] tag rows: {tag_table.num_rows:,}")
    size = sum(p.stat().st_size for p in LB.glob("*.parquet")) / 2**20
    print(f"[extract] wrote {size:.0f} MB to data/lb/ in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    run()
