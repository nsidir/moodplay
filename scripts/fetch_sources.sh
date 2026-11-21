#!/usr/bin/env bash
# Fetch the extra open datasets for musicdb (resumable; writes <file>.done when a file is complete).
# usage: scripts/fetch_sources.sh [dest dir, default data/raw/downloads]   (or: make fetch)
# Sources: ListenBrainz full feedback dump (CC0), MSD track list + Echo Nest Taste Profile (research use),
#          AcousticBrainz low-level features + high-level classifiers (CC0). About 45 GB in total.
set -u
DEST="${1:-data/raw/downloads}"; mkdir -p "$DEST"; cd "$DEST"
MB=https://data.metabrainz.org/pub/musicbrainz
AB=$MB/acousticbrainz/dumps
# newest full feedback dump (they rotate, so look it up)
LBD=$(curl -s --max-time 60 "$MB/listenbrainz/spark/" | grep -oE 'listenbrainz-feedback-[0-9]+-[0-9]+-full' | sort -u | tail -1)
[ -n "$LBD" ] || { echo "could not list $MB/listenbrainz/spark/"; exit 1; }
LBF=$MB/listenbrainz/spark/$LBD/listenbrainz-feedback-dump-${LBD#listenbrainz-feedback-}
LBF=${LBF%-full}.tar.zst
MSD=http://millionsongdataset.com/sites/default/files
urls=("$LBF" "$LBF.sha256" "$MSD/challenge/train_triplets.txt.zip" "$MSD/AdditionalFiles/unique_tracks.txt"
      "$MSD/tasteprofile/sid_mismatches.txt"
      "$AB/acousticbrainz-lowlevel-features-20220623/sha256sums"
      "$AB/acousticbrainz-highlevel-json-20220623/sha256sums")
for k in rhythm lowlevel tonal; do urls+=("$AB/acousticbrainz-lowlevel-features-20220623/acousticbrainz-lowlevel-features-20220623-$k.tar.zst"); done
for i in $(seq 0 29); do urls+=("$AB/acousticbrainz-highlevel-json-20220623/acousticbrainz-highlevel-json-20220623-$i.tar.zst"); done
for u in "${urls[@]}"; do
  f=$(basename "$u")
  case "$u" in */acousticbrainz-lowlevel-features-20220623/sha256sums) f=ab-lowlevel-features.sha256sums;; */acousticbrainz-highlevel-json-20220623/sha256sums) f=ab-highlevel.sha256sums;; esac
  { [ -e "$f.done" ] || [ -e "$f.ingested" ]; } && continue   # downloaded, or already in musicdb
  for attempt in 1 2 3 4 5 6 7 8; do
    if curl -sS -fL -C - --speed-limit 10000 --speed-time 60 --connect-timeout 30 --retry 5 --retry-delay 10 -o "$f" "$u"; then touch "$f.done"; echo "done $f"; break; fi
    echo "retry $attempt $f"; sleep 20
  done
done
echo ALLDONE
