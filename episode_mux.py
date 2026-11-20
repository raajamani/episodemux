#!/usr/bin/env python3
"""
Episode Mux — auto-mux a whole season (anime or regular TV) from one
verified mkvmerge command.

Workflow:
  1. Mux ONE episode in MKVToolNix GUI exactly how you want it.
  2. Multiplexer -> "Copy command line to clipboard".
  3. Paste into COMMAND (main episodes) and/or COMMAND_SPECIALS below.
  4. Run this script.

For each COMMAND the script:
  - Reads the source folders / file paths directly from the command line
    (no folder sniffing — what you pasted is the ground truth).
  - Loops through all matching episode files in those exact folders.
  - Auto-detects sync offset via audio cross-correlation; clamps insane
    results (> SANITY_MS) to 0ms; falls back to 0ms if correlation fails.
  - Never skips an episode.

Leave COMMAND_SPECIALS = "" to skip the specials pass entirely.

Outputs:
  <base-folder>/_muxed/          <- main episodes
  <base-folder>/_muxed_specials/ <- specials (only if COMMAND_SPECIALS set)
"""
import subprocess, os, sys, json, re, shlex, tempfile, statistics
import numpy as np
from scipy.io import wavfile
from scipy.signal import correlate

# ============================================================ PASTE HERE
COMMAND = r"""
/Applications/MKVToolNix.app/Contents/MacOS/mkvmerge --ui-language en_US --priority lower --output '/Volumes/3.34TB_SMR/Media/TV Series/Star Wars Tales of the Underworld/Star.Wars.Tales.of.the.Underworld.S01E01.MULTi.2160p.DSNP.WEB-DL.DDP5.1.DV.HDR.H.265-P2P.mkv' --audio-tracks 2 --subtitle-tracks 5 --no-global-tags --language 0:en --display-dimensions 0:3840x2160 --language 2:en --track-name 2:English --sub-charset 5:UTF-8 --language 5:en --track-name 5:English '(' '/Volumes/3.34TB_SMR/Media/TV Series/Star Wars Tales of the Underworld/S01E01.mkv' ')' --title Star.Wars.Tales.of.the.Underworld.S01E01.MULTi.2160p.DSNP.WEB-DL.DDP5.1.DV.HDR.H.265-P2P --track-order 0:0,0:2,0:5
"""

# Paste your specials mux command here when ready. Leave blank to skip.
COMMAND_SPECIALS = r"""
"""

# ============================================================ CONFIG
FFMPEG = "ffmpeg"

# Title is derived automatically from the parent folder of the base source.
# e.g. if base is .../Why the Hell are You Here, Teacher!?/[Moozzi2].../ep.mkv
#      title becomes "Why the Hell are You Here, Teacher!?"
# Override here only if you want something different (leave blank = auto).
TITLE_OVERRIDE = ""

VIDEO_PRIORITY = ["Moozzi2", "VCB"]
KNOWN_GROUPS   = ["Moozzi2", "VCB", "iAHD", "EMBER", "Judas", "DiabloTripleA", "Tenrai-Sensei", "Tenrai"]

SR                 = 8000
DRIFT_TOLERANCE_MS = 60
CROSS_TOLERANCE_MS = 80
SANITY_MS          = 30000  # clamp only truly impossible offsets (corrupt correlation)


# Per-episode subtitle fine-tune (ms). Only needed for broken encodes like EMBER
# where the ASS subs are not in sync with the audio/video.
# Positive = subs arrive late (pull them earlier).
# Negative = subs arrive early (push them later).
# Clear this dict {} when switching to a well-encoded source.
SUB_TRIM_MS = {}

# Convert any TrueHD track being muxed to lossless FLAC (same channels,
# same bit depth). e.g. TrueHD 5.1 24-bit -> FLAC 5.1 24-bit.
CONVERT_TRUEHD_TO_FLAC = True
FLAC_COMPRESSION       = 8      # 0-12, 8 = good size/speed balance
FFPROBE                = "ffprobe"
# ====================================================================

NOISE = re.compile(
    r'(2160p?|1080p?|720p?|480p?|x?26[45]|hevc|avc|10.?bit|8.?bit|bdrip|blu.?ray|'
    r'web.?dl|flac|dts(?:-?hd)?|e?ac3|aac|ddp?\d?|dual.?audio|\b(?:19|20)\d{2}\b|'
    r'\[[0-9A-Fa-f]{8}\])', re.I)

DUP_SUFFIX = re.compile(r'\s*\(\d+\)$')


def detect_group(text):
    for g in KNOWN_GROUPS:
        if re.search(re.escape(g), text, re.I):
            return g
    return None


def episode_number(filename):
    name = DUP_SUFFIX.sub('', os.path.splitext(filename)[0])
    name = NOISE.sub(' ', name)
    m = re.search(r'[Ss]\d{1,2}[\s._-]*[Ee](\d{1,3})', name)
    if m: return int(m.group(1))
    m = re.search(r'\b(?:E|EP|episode)[\s._-]*(\d{1,3})\b', name, re.I)
    if m: return int(m.group(1))
    m = re.search(r'-\s*(\d{1,3})(?=\D|$)', name)
    if m: return int(m.group(1))
    nums = re.findall(r'\b(\d{1,3})\b', name)
    return int(nums[-1]) if nums else None


def is_special(filename):
    return bool(re.search(r'\b(SP|OVA|OAD|NCED|NCOP|Special|S\d{1,2}SP)\b',
                           filename, re.I))


def list_episodes(folder, required=None, allow_specials=False):
    """
    Scan folder (root only) for MKV/MP4.
    allow_specials=False -> skip files that look like specials.
    required -> filename must contain this string (case-insensitive).
    Returns {episode_number: full_path}.

    If episode_number() fails for every file (unusual naming like S01SP01),
    falls back to positional numbering by sort order (1, 2, 3 ...).
    """
    candidates = []
    try:
        entries = sorted(os.listdir(folder))
    except PermissionError:
        return {}
    for fname in entries:
        full = os.path.join(folder, fname)
        if os.path.isdir(full):
            continue
        if not fname.lower().endswith((".mkv", ".mp4")):
            continue
        if required and required.lower() not in fname.lower():
            continue
        if not allow_specials and is_special(fname):
            continue
        candidates.append((fname, full))

    # First pass: try to extract episode numbers normally
    eps = {}
    for fname, full in candidates:
        n = episode_number(fname)
        if n is not None and n not in eps:
            eps[n] = full

    # If nothing parsed (e.g. S01SP01 naming), fall back to positional order
    if not eps and candidates:
        print(f"    [positional fallback] no episode numbers parsed in {folder!r}")
        eps = {i + 1: full for i, (_, full) in enumerate(candidates)}

    return eps


def identify(mkvmerge, path):
    out = subprocess.run([mkvmerge, "-J", path], capture_output=True, text=True)
    return json.loads(out.stdout)



def ass_track_index_in_file(mkvmerge, path, track_id):
    """
    Return which index (0-based) track_id is among ASS tracks in path.
    e.g. if the file has ASS tracks at ids [1, 3, 5] and track_id=3, returns 1.
    Returns 0 if not found.
    """
    info = identify(mkvmerge, path)
    ass_tracks = []
    for t in info["tracks"]:
        if t["type"] != "subtitles":
            continue
        codec = (t.get("codec") or "").lower()
        if "substationalpha" in codec or "ssa" in codec or "ass" in codec:
            ass_tracks.append(t["id"])
    try:
        return ass_tracks.index(track_id)
    except ValueError:
        return 0


def find_ass_track_id(mkvmerge, path, ass_idx=0):
    """
    Return the track ID of the Nth (0-based) ASS track in path.
    ass_idx comes from the template file so the user's GUI choice is preserved.
    """
    info = identify(mkvmerge, path)
    ass_tracks = []
    for t in info["tracks"]:
        if t["type"] != "subtitles":
            continue
        codec = (t.get("codec") or "").lower()
        if "substationalpha" in codec or "ssa" in codec or "ass" in codec:
            ass_tracks.append(t["id"])
    if not ass_tracks:
        return None
    idx = min(ass_idx, len(ass_tracks) - 1)
    return ass_tracks[idx]

def jp_audio_index(info):
    ai, found = 0, None
    for t in info["tracks"]:
        if t["type"] == "audio":
            lang = (t.get("properties", {}).get("language") or "").lower()
            if lang in ("jpn", "ja", "jap") and found is None:
                found = ai
            ai += 1
    return found if found is not None else 0


def duration_s(info):
    try:
        return info["container"]["properties"]["duration"] / 1e9
    except Exception:
        return 1440.0


def extract(path, a_index, start, dur):
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name
    subprocess.run(
        [FFMPEG, "-y", "-ss", str(start), "-t", str(dur), "-i", path,
         "-map", f"0:a:{a_index}", "-ac", "1", "-ar", str(SR), "-vn", tmp],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _, data = wavfile.read(tmp)
    os.remove(tmp)
    return data.astype(np.float32)


def offset_ms(ref, ref_i, tgt, tgt_i, dur):
    # Scale extraction window to file duration so short clips work too.
    # Use up to 60s but never more than 40% of the file, minimum 5s.
    win = max(5.0, min(60.0, dur * 0.40))
    min_samples = SR * 3  # need at least 3s of audio to correlate meaningfully

    # Sample at 25%, 50%, 72% — but only if there's enough room either side.
    vals = []
    for frac in (0.25, 0.50, 0.72):
        s = dur * frac
        if s < 2 or s + win > dur:
            s = max(2.0, dur - win - 2)   # fall back to near the end
        try:
            r = extract(ref, ref_i, s, win)
            t = extract(tgt, tgt_i, s, win)
        except Exception:
            continue
        n = min(len(r), len(t))
        if n < min_samples:
            continue
        r, t = r[:n] - r[:n].mean(), t[:n] - t[:n].mean()
        corr = correlate(t, r, mode="full", method="fft")
        vals.append((np.argmax(corr) - (n - 1)) / SR * 1000.0)
    if not vals:
        return None, 0
    return statistics.median(vals), max(vals) - min(vals)


def ass_time_to_ms(t_str):
    """Convert ASS timestamp h:mm:ss.cc to milliseconds."""
    t_str = t_str.strip()
    h, m, rest = t_str.split(":")
    s, cs = rest.split(".")
    return (int(h)*3600 + int(m)*60 + int(s)) * 1000 + int(cs) * 10


def extract_ass_events(mkvmerge, mkv_path):
    """
    Extract ASS track from MKV, return list of (start_ms, text) tuples.
    text is stripped of ASS override tags for comparison.
    """
    info = identify(mkvmerge, mkv_path)
    ass_tid = None
    for t in info["tracks"]:
        if t["type"] != "subtitles":
            continue
        codec = (t.get("codec") or "").lower()
        if "substationalpha" in codec or "ssa" in codec or "ass" in codec:
            ass_tid = t["id"]
            break
    if ass_tid is None:
        return []

    tmp = tempfile.NamedTemporaryFile(suffix=".ass", delete=False).name
    try:
        subprocess.run(
            ["mkvextract", mkv_path, "tracks", f"{ass_tid}:{tmp}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        os.remove(tmp)
        return []

    events = []
    tag_re = re.compile(r'{[^}]*}')
    try:
        with open(tmp, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("Dialogue:"):
                    continue
                parts = line.split(",", 9)
                if len(parts) < 10:
                    continue
                try:
                    start_ms = ass_time_to_ms(parts[1])
                except Exception:
                    continue
                text = tag_re.sub("", parts[9]).replace(r"\N", " ").strip()
                if text:
                    events.append((start_ms, text))
    finally:
        os.remove(tmp)

    return events


def extract_pgs_timestamps(mkvmerge, mkv_path, track_name_hint="Dialogue"):
    """
    Return PGS subtitle start times (ms) for a named track.
    Uses mkvmerge -J timestamp data — PGS track timestamps from container.
    Falls back to first PGS track if name hint not found.
    """
    info = identify(mkvmerge, mkv_path)
    target_tid = None
    fallback_tid = None
    for t in info["tracks"]:
        if t["type"] != "subtitles":
            continue
        codec = (t.get("codec") or "").lower()
        if "pgs" not in codec and "hdmv" not in codec:
            continue
        name = (t.get("properties", {}).get("track_name") or "").lower()
        if fallback_tid is None:
            fallback_tid = t["id"]
        if track_name_hint.lower() in name:
            target_tid = t["id"]
            break
    tid = target_tid if target_tid is not None else fallback_tid
    if tid is None:
        return []

    # Extract PGS to a temp file, read timestamps via mkvmerge -J on it
    tmp = tempfile.NamedTemporaryFile(suffix=".sup", delete=False).name
    try:
        subprocess.run(
            ["mkvextract", mkv_path, "tracks", f"{tid}:{tmp}"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        out = subprocess.run(
            ["mkvmerge", "-J", tmp],
            capture_output=True, text=True)
        # mkvmerge -J on a .sup doesn't give timestamps; use ffprobe instead
        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-select_streams", "s:0",
             "-show_entries", "packet=pts_time",
             "-of", "csv=p=0", tmp],
            capture_output=True, text=True)
        timestamps = []
        for line in probe.stdout.splitlines():
            line = line.strip()
            if line and line != "N/A":
                try:
                    timestamps.append(round(float(line) * 1000))
                except ValueError:
                    continue
        return timestamps
    except Exception:
        return []
    finally:
        os.remove(tmp)


def sub_offset_ms_by_text(mkvmerge, ass_mkv, ref_mkv):
    """
    Compare EMBER ASS subtitle timestamps against iAHD audio-correlation-derived
    timestamps by using the video-shift-corrected audio offset.

    Better approach: extract ASS events and PGS dialogue timestamps from ref,
    find the median time delta between matching lines.
    Returns (offset_ms, confidence) or (None, 0).
    """
    ass_events = extract_ass_events(mkvmerge, ass_mkv)
    ref_pgs    = extract_pgs_timestamps(mkvmerge, ref_mkv, "Dialogue")

    if not ass_events or not ref_pgs:
        return None, 0

    # ref_pgs is a list of timestamps. ass_events is (ms, text).
    # Strategy: for each ASS event, find the nearest PGS timestamp and
    # compute the delta. Use the median of all deltas as the offset.
    # Filter out outliers (more than 2s from median).
    ass_times = [ms for ms, _ in ass_events if ms > 5000]  # skip OP
    ref_times = sorted(ref_pgs)

    if not ass_times or not ref_times:
        return None, 0

    deltas = []
    for a_ms in ass_times[:30]:
        # find closest ref timestamp
        closest = min(ref_times, key=lambda r: abs(r - a_ms))
        delta = a_ms - closest
        if abs(delta) < 5000:   # ignore if more than 5s apart (different line)
            deltas.append(delta)

    if len(deltas) < 3:
        return None, 0

    med = statistics.median(deltas)
    # Filter to deltas within 500ms of median
    filtered = [d for d in deltas if abs(d - med) < 500]
    if not filtered:
        return None, 0

    spread = max(filtered) - min(filtered)
    return round(statistics.median(filtered)), spread



# --------------------------------------------------- TrueHD -> FLAC
# Per-track options that must follow the audio track to its new FLAC file.
TRACK_OPTS = {"--language", "--track-name", "--default-track-flag",
              "--forced-display-flag", "--track-enabled-flag", "--sync",
              "--commentary-flag", "--original-flag", "--visual-impaired-flag",
              "--hearing-impaired-flag", "--text-descriptions-flag", "--cues",
              "--compression"}


def _kept_audio_ids(cmd, start, end, info):
    """Audio track ids from this source block that mkvmerge will actually mux."""
    all_audio = [t["id"] for t in info["tracks"] if t["type"] == "audio"]
    for k in range(start, end):
        if cmd[k] in ("--no-audio", "-A"):
            return []
        if cmd[k] in ("--audio-tracks", "-a") and k + 1 < end:
            spec = cmd[k + 1]
            if spec.startswith("!"):
                excl = {int(x) for x in spec[1:].split(",") if x.strip().isdigit()}
                return [i for i in all_audio if i not in excl]
            return [int(x) for x in spec.split(",") if x.strip().isdigit()]
    return all_audio


def _bit_depth(path, tid):
    try:
        out = subprocess.run(
            [FFPROBE, "-v", "error", "-select_streams", str(tid),
             "-show_entries", "stream=bits_per_raw_sample", "-of", "json", path],
            capture_output=True, text=True)
        b = int(json.loads(out.stdout)["streams"][0].get("bits_per_raw_sample") or 0)
        return 16 if b and b <= 16 else 24
    except Exception:
        return 24


def _to_flac(path, tid, dst):
    bits = _bit_depth(path, tid)
    fmt  = "s16" if bits == 16 else "s32"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error", "-i", path, "-map", f"0:{tid}",
         "-c:a", "flac", "-compression_level", str(FLAC_COMPRESSION),
         "-sample_fmt", fmt, "-bits_per_raw_sample", str(bits), dst],
        check=True)
    return bits


_ATMOS_CHOICE = None   # remembered for the whole run once answered with "all"


def _ask_atmos(fname, tid):
    """Warn about Atmos and ask whether to convert. Returns True to convert."""
    global _ATMOS_CHOICE
    if _ATMOS_CHOICE is not None:
        return _ATMOS_CHOICE
    print()
    print("  " + "!" * 62)
    print(f"  WARNING: TrueHD Atmos found (track {tid}) in:")
    print(f"    {fname}")
    print("  Converting to FLAC keeps the 5.1/7.1 bed losslessly but")
    print("  PERMANENTLY DISCARDS the Atmos object/height metadata.")
    print("  " + "!" * 62)
    if not sys.stdin.isatty():
        print("  (no terminal to ask -- keeping TrueHD Atmos)")
        _ATMOS_CHOICE = False
        return False
    while True:
        ans = input("  Convert to FLAC?  [y]es / [n]o / [Y]es to all / [N]o to all: ").strip()
        if ans in ("y", "yes"):  return True
        if ans in ("n", "no", ""): return False
        if ans == "Y": _ATMOS_CHOICE = True;  return True
        if ans == "N": _ATMOS_CHOICE = False; return False
        print("  Please answer y, n, Y or N.")


def convert_truehd_tracks(mkvmerge, cmd, tmpdir, label):
    """
    Find every TrueHD track the command will mux, encode it to FLAC in
    tmpdir, drop the TrueHD from its source, and add the FLAC as a new
    source carrying the same language/name/flags/sync. --track-order is
    rewritten so the FLAC sits exactly where the TrueHD was.
    Returns the new cmd list.
    """
    _, _, files = parse_command(cmd)
    jobs = []   # (file_index, block, tid)
    for fi, f in enumerate(files):
        path = cmd[f["path_idx"]]
        info = identify(mkvmerge, path)
        codecs = {t["id"]: (t.get("codec") or "") for t in info["tracks"]}
        start = f["opt_start"]
        end   = f["paren_open"] if f["paren_open"] is not None else f["path_idx"]
        for tid in _kept_audio_ids(cmd, start, end, info):
            codec = codecs.get(tid, "").lower()
            if "truehd" not in codec:
                continue
            if "atmos" in codec and not _ask_atmos(os.path.basename(path), tid):
                print(f"    {label}: keeping TrueHD Atmos tid={tid} as-is")
                continue
            jobs.append((fi, f, tid, info))
    if not jobs:
        return cmd

    n_files = len(files)
    new_blocks, order_map, removals = [], {}, {}
    for n, (fi, f, tid, info) in enumerate(jobs):
        path = cmd[f["path_idx"]]
        dst  = os.path.join(tmpdir, f"f{fi}_t{tid}.flac")
        ch   = next((t.get("properties", {}).get("audio_channels")
                     for t in info["tracks"] if t["id"] == tid), "?")
        print(f"    {label}: TrueHD tid={tid} ({ch}ch) -> FLAC ...", end="", flush=True)
        bits = _to_flac(path, tid, dst)
        print(f" done ({bits}-bit)")

        start = f["opt_start"]
        end   = f["paren_open"] if f["paren_open"] is not None else f["path_idx"]
        opts  = []
        for k in range(start, end - 1):
            if cmd[k] in TRACK_OPTS and cmd[k + 1].split(":", 1)[0] == str(tid):
                opts += [cmd[k], "0:" + cmd[k + 1].split(":", 1)[1]]
                removals.setdefault(fi, set()).update({k, k + 1})
        new_blocks.append(opts + ["(", dst, ")"])
        order_map[f"{fi}:{tid}"] = f"{n_files + n}:0"
        removals.setdefault(fi, set())

    # Exclude the TrueHD tracks from their original sources (right-to-left).
    by_file = {}
    for fi, f, tid, _ in jobs:
        by_file.setdefault(fi, (f, []))[1].append(tid)
    edits = []   # (index, action, payload)
    for fi, (f, tids) in by_file.items():
        start = f["opt_start"]
        end   = f["paren_open"] if f["paren_open"] is not None else f["path_idx"]
        handled = False
        for k in range(start, end):
            if cmd[k] in ("--audio-tracks", "-a") and k + 1 < end:
                spec = cmd[k + 1]
                if spec.startswith("!"):
                    cmd[k + 1] = spec + "," + ",".join(map(str, tids))
                else:
                    keep = [x for x in spec.split(",") if x not in map(str, tids)]
                    if keep:
                        cmd[k + 1] = ",".join(keep)
                    else:
                        cmd[k], cmd[k + 1] = "--no-audio", None
                handled = True
                break
        if not handled:
            edits.append((end, ["--audio-tracks", "!" + ",".join(map(str, tids))]))

    drop = set().union(*removals.values()) if removals else set()
    for idx, toks in sorted(edits, key=lambda e: e[0], reverse=True):
        cmd[idx:idx] = toks
        drop = {d + len(toks) if d >= idx else d for d in drop}
    cmd = [t for i, t in enumerate(cmd) if i not in drop and t is not None]

    # Rewrite --track-order, then append the FLAC sources at the very end.
    if "--track-order" in cmd:
        k = cmd.index("--track-order") + 1
        cmd[k] = ",".join(order_map.get(x, x) for x in cmd[k].split(","))
    for b in new_blocks:
        cmd += b
    return cmd

# --------------------------------------------------- command parsing
def parse_command(tokens):
    """
    Returns (mkvmerge_bin, out_idx, files).
    Each entry in files:
      path_idx              – index of the source file path in tokens
      opt_start/paren_open/paren_close – token indices for option blocks
    """
    out_idx     = tokens.index("--output") + 1
    files       = []
    first_paren = next((i for i, t in enumerate(tokens) if t == "("), len(tokens))

    # Bare source before the first '('
    bare_found = False
    for i in range(out_idx + 1, first_paren):
        tok  = tokens[i]
        prev = tokens[i - 1] if i > 0 else ""
        if tok.startswith("--") or prev.startswith("--"):
            continue
        if os.sep in tok or tok.lower().endswith((".mkv", ".mp4", ".avi")):
            files.append({
                "opt_start":   out_idx + 1,
                "paren_open":  None,
                "path_idx":    i,
                "paren_close": None,
            })
            bare_found = True
            break

    # Paren-enclosed sources
    # If no bare source was found, any options before the first '(' belong
    # to the first paren-enclosed source (mkvmerge allows global options
    # before the first source block).
    pre_paren_opt_start = out_idx + 1 if not bare_found else first_paren
    i = block_start = first_paren
    first_paren_source = True
    while i < len(tokens):
        if tokens[i] == "(":
            j, paths = i + 1, []
            while j < len(tokens) and tokens[j] != ")":
                paths.append(j)
                j += 1
            if paths:
                # Give pre-paren options to the first paren source if no bare source
                opt_s = pre_paren_opt_start if (first_paren_source and not bare_found) else block_start
                files.append({
                    "opt_start":   opt_s,
                    "paren_open":  i,
                    "path_idx":    paths[0],
                    "paren_close": j,
                })
                first_paren_source = False
            i = block_start = j + 1
        else:
            i += 1

    return tokens[0], out_idx, files


# --------------------------------------------------- mux pass
def run_pass(mkvmerge, tokens, out_idx, files, base, season,
             ep_maps, common, out_dir, title, ep_label_fmt, allow_specials=False):
    os.makedirs(out_dir, exist_ok=True)

    # For each non-base source, find which ASS-track-index the template
    # command's --subtitle-tracks value refers to. This preserves the user's
    # GUI choice (e.g. 3rd ASS track) across all episodes.
    source_ass_idx = {}  # path_idx -> ass_idx (0-based among ASS tracks)
    for f in files:
        if f is base or f["paren_open"] is None:
            continue
        start_k = f["opt_start"]
        end_k   = f["paren_open"]
        for k in range(start_k, end_k):
            if tokens[k] == "--subtitle-tracks" and k + 1 < end_k:
                template_tid = int(tokens[k + 1].split(',')[0])
                template_file = tokens[f["path_idx"]]
                source_ass_idx[f["path_idx"]] = ass_track_index_in_file(
                    mkvmerge, template_file, template_tid)
                print(f"  [{f['group'] or 'unknown'}] template --subtitle-tracks {template_tid} "
                      f"-> ASS track index {source_ass_idx[f['path_idx']]}")
                break

    for ep in sorted(common):
        base_file = ep_maps[base["path_idx"]][ep]
        binfo     = identify(mkvmerge, base_file)
        bdur, bjp = duration_s(binfo), jp_audio_index(binfo)

        # Measure offset independently for each non-base source.
        # For the video shift we use the REFERENCE source (highest-priority
        # non-base that has clean audio, e.g. iAHD). For sources whose audio
        # may be misaligned from their own subs (e.g. EMBER), we ALSO measure
        # that source against the reference to get a sub-specific correction.
        src_shifts = {}   # path_idx -> audio offset vs Moozzi2 (used for video shift)
        label = ep_label_fmt.format(n=ep, season=season)

        # Identify the best reference source (non-base, highest priority group)
        non_base = [f for f in files if f is not base]

        for f in non_base:
            tgt  = ep_maps[f["path_idx"]][ep]
            info = identify(mkvmerge, tgt)
            off, spread = offset_ms(base_file, bjp, tgt, jp_audio_index(info), bdur)
            if off is None:
                src_shifts[f["path_idx"]] = 0
                print(f"  {label} [{f['group'] or 'unknown'}]: correlation failed -> 0ms")
            else:
                raw = round(off)
                if abs(raw) > SANITY_MS:
                    src_shifts[f["path_idx"]] = 0
                    print(f"  {label} [{f['group'] or 'unknown'}]: insane offset {raw:+d}ms -> 0ms")
                else:
                    src_shifts[f["path_idx"]] = raw
                    warn = "  [drift]" if spread > DRIFT_TOLERANCE_MS else ""
                    print(f"  {label} [{f['group'] or 'unknown'}]: "
                          f"source offset {raw:+d}ms{warn}")

        ep_name      = f"{title} {label}"
        cmd          = list(tokens)
        cmd[out_idx] = os.path.join(out_dir, ep_name + ".mkv")

        if "--title" in cmd:
            cmd[cmd.index("--title") + 1] = ep_name

        for f in files:
            cmd[f["path_idx"]] = ep_maps[f["path_idx"]][ep]

        for f in files:
            if f is base:
                continue
            if f["paren_open"] is None:
                continue
            ep_path = ep_maps[f["path_idx"]][ep]
            start_k = f["opt_start"]
            end_k   = f["paren_open"]
            for k in range(start_k, end_k):
                if cmd[k] == "--subtitle-tracks" and k + 1 < end_k:
                    real_tid = find_ass_track_id(mkvmerge, ep_path, ass_idx=source_ass_idx.get(f['path_idx'], 0))
                    if real_tid is not None:
                        old_ids = cmd[k + 1].split(",")
                        old_tid = old_ids[0]
                        cmd[k + 1] = ",".join([str(real_tid)] + old_ids[1:])
                        for j in range(k + 2, end_k):
                            if ":" in cmd[j]:
                                parts = cmd[j].split(":", 1)
                                if parts[0] == old_tid:
                                    cmd[j] = f"{real_tid}:{parts[1]}"
                    break


        # Base (video+audio) stays untouched -- its A/V is already in sync.
        # Instead, shift every subtitle track pulled from a non-base source
        # by that source's measured audio offset (e.g. -1001ms -> --sync +1001).
        # Process blocks right-to-left so inserts don't move later indices.
        for f in sorted(files, key=lambda x: x["paren_open"] or -1, reverse=True):
            if f is base or f["paren_open"] is None:
                continue
            shift = -src_shifts.get(f["path_idx"], 0)
            sub_ids = []
            for k in range(f["opt_start"], f["paren_open"]):
                if cmd[k] == "--subtitle-tracks":
                    sub_ids = cmd[k + 1].split(",")
                    break
            if shift == 0 or not sub_ids:
                continue
            for tid in sub_ids:
                cmd.insert(f["paren_open"], f"{tid}:{shift}")
                cmd.insert(f["paren_open"], "--sync")
            print(f"    [{f['group'] or 'unknown'}] subs {sub_ids} -> --sync {shift:+d}ms")

        flac_tmp = tempfile.mkdtemp(prefix="truehd2flac_") if CONVERT_TRUEHD_TO_FLAC else None
        try:
            if flac_tmp:
                cmd = convert_truehd_tracks(mkvmerge, cmd, flac_tmp, label)
            result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        finally:
            if flac_tmp:
                import shutil
                shutil.rmtree(flac_tmp, ignore_errors=True)
        if result.returncode != 0:
            print(f"  ERROR on {label}: mkvmerge returned {result.returncode}")
            print(f"  stderr: {result.stderr.decode(errors='replace').strip()}")
            print(f"  cmd: {' '.join(cmd)}")
            raise SystemExit(1)

    print(f"  Done -> {out_dir}\n")


# --------------------------------------------------- single pass setup
def setup_pass(command_str, pass_name, out_suffix, ep_label_fmt,
               allow_specials=False):
    """
    Parse command_str, scan the folders it references, and run the mux.
    Folders and filenames come entirely from the command — no sniffing.
    """
    print("=" * 60)
    print(f"PASS: {pass_name}")
    print("=" * 60)

    tokens   = shlex.split(command_str.strip())
    mkvmerge, out_idx, files = parse_command(tokens)

    if not files:
        print("  ERROR: no source files found in command.")
        return

    for f in files:
        path        = tokens[f["path_idx"]]
        f["group"]  = detect_group(path)
        f["folder"] = os.path.dirname(path)

    def prio(g):
        return VIDEO_PRIORITY.index(g) if g in VIDEO_PRIORITY else 99

    base = min(files, key=lambda f: prio(f["group"]))

    print(f"  base:   {base['group']}  ->  {base['folder']}")
    for f in files:
        if f is not base:
            print(f"  source: {f['group'] or 'unknown'}  ->  {f['folder']}")

    # Folder check
    print()
    all_ok = True
    for f in files:
        folder  = f["folder"]
        exists  = os.path.exists(folder)
        is_dir  = os.path.isdir(folder) if exists else False
        try:
            entries  = os.listdir(folder) if is_dir else []
            readable = True
        except Exception:
            entries, readable = [], False
        status = "OK" if (exists and is_dir and readable) else "PROBLEM"
        if status == "PROBLEM":
            all_ok = False
        mkv_count = sum(1 for e in entries if e.lower().endswith((".mkv", ".mp4")))
        print(f"  [{f['group'] or 'unknown'}] {status}: "
              f"{len(entries)} entries, {mkv_count} video files  |  {folder!r}")
        if status == "PROBLEM":
            parent = os.path.dirname(folder)
            if os.path.isdir(parent):
                print(f"    -> parent: {sorted(os.listdir(parent))}")

    if not all_ok:
        print("  One or more folders unreadable — make available offline and retry.")
        return
    print()

    # Required-group filter: only apply if the template filename contains the group tag
    for f in files:
        base_name = os.path.basename(tokens[f["path_idx"]])
        f["required"] = (f["group"]
                         if f["group"] and f["group"].lower() in base_name.lower()
                         else None)

    ep_maps = {
        f["path_idx"]: list_episodes(f["folder"], f["required"], allow_specials)
        for f in files
    }

    for f in files:
        print(f"  [{f['group'] or 'unknown'}] found: "
              f"{sorted(ep_maps[f['path_idx']].keys())}")

    common = set(ep_maps[base["path_idx"]])
    for f in files:
        common &= set(ep_maps[f["path_idx"]])

    if not common:
        print("  No common episodes found — check detection above.")
        return

    # Folder layout:
    #   release-group layout: Show/[Group] Release/ep.mkv -> title "Show",
    #                         output Show/_muxed/
    #   flat layout (TV):     Show/S01E01.mkv            -> title "Show",
    #                         output Show/_muxed/
    base_dir_name = os.path.basename(base["folder"])
    grouped = bool(detect_group(base_dir_name) or re.match(r"^\s*\[", base_dir_name))
    show_dir = os.path.dirname(base["folder"]) if grouped else base["folder"]
    if TITLE_OVERRIDE.strip():
        title = TITLE_OVERRIDE.strip()
    else:
        title = os.path.basename(show_dir) or base_dir_name
    out_dir = os.path.join(show_dir, out_suffix)
    print(f"  layout:  {'release-group folders' if grouped else 'flat (TV)'}")

    print(f"  title:   {title!r}")
    print(f"  output:  {out_dir}")
    print(f"  to mux:  {sorted(common)}\n")

    m = re.search(r'[Ss](\d{1,2})[\s._-]*[Ee]\d{1,3}', os.path.basename(tokens[base["path_idx"]]))
    season = int(m.group(1)) if m else None
    if season is not None and ep_label_fmt.startswith("E"):
        ep_label_fmt = "S{season:02d}" + ep_label_fmt
    print(f"  labels:  {ep_label_fmt.format(n=1, season=season or 0)} ...")

    run_pass(mkvmerge, tokens, out_idx, files, base, season,
             ep_maps, common, out_dir, title, ep_label_fmt, allow_specials)


# --------------------------------------------------- main
def main():
    if COMMAND.strip():
        setup_pass(COMMAND, "main episodes", "_muxed",
                   ep_label_fmt="E{n:02d}", allow_specials=False)

    if COMMAND_SPECIALS.strip():
        setup_pass(COMMAND_SPECIALS, "specials", "_muxed_specials",
                   ep_label_fmt="SP{n:02d}", allow_specials=True)

    print("All done.")


if __name__ == "__main__":
    main()
