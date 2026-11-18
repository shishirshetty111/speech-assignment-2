"""Shared paths and a helper to run MFA commands."""
import os
import shutil
import subprocess
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")  # IPA phones in the english_mfa dictionary

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
CORPUS = os.path.join(ROOT, "data", "corpus")          # MFA-format corpus (generated)
WORK = os.path.join(ROOT, "work")                      # MFA temporary files and trained models (generated)
DICTS = os.path.join(ROOT, "dictionaries")
OUT = os.path.join(ROOT, "outputs")
TEXTGRIDS = os.path.join(OUT, "textgrids")
FIGURES = os.path.join(OUT, "figures")
ANALYSIS = os.path.join(OUT, "analysis")


def praat_executable():
    """Praat: the PRAAT environment variable, else praat on PATH, else the local install folder."""
    for cand in (os.environ.get("PRAAT"), shutil.which("praat"), shutil.which("Praat"),
                 os.path.join("D:" + os.sep, "tools", "praat", "Praat.exe")):
        if cand and os.path.exists(cand):
            return cand
    sys.exit("Praat not found: install it from https://www.praat.org and set PRAAT=<path to the executable>")


def speaker_of(utt_id):
    """Speaker folder for a file: the BU Radio News files start with the speaker code (F2B),
    the ISLE files carry the session id."""
    if utt_id.upper().startswith("ISLE_"):
        return "isle_" + utt_id.split("_")[1].lower()
    return utt_id[:3].lower()


def _env():
    """Make the conda environment's DLL folders visible when Python is started directly
    from the environment without activating it."""
    env = dict(os.environ)
    extra = [sys.prefix, os.path.join(sys.prefix, "Library", "bin"), os.path.join(sys.prefix, "Scripts"),
             os.path.join(sys.prefix, "bin")]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def mfa_executable():
    for cand in (os.path.join(sys.prefix, "Scripts", "mfa.exe"), os.path.join(sys.prefix, "bin", "mfa")):
        if os.path.exists(cand):
            return cand
    found = shutil.which("mfa")
    if not found:
        sys.exit("mfa not found: run this with the Python of the MFA conda environment (see README)")
    return found


def run_mfa(*args, log=None):
    """Run `mfa <args>`; stop the pipeline if it fails."""
    cmd = [mfa_executable(), *map(str, args)]
    print(">> mfa " + " ".join(map(str, args)), flush=True)
    res = subprocess.run(cmd, env=_env(), capture_output=True, text=True, encoding="utf-8", errors="replace")
    if log:
        os.makedirs(os.path.dirname(log), exist_ok=True)
        with open(log, "w", encoding="utf-8") as f:
            f.write(res.stdout + "\n" + res.stderr)
    if res.returncode != 0:
        print(res.stdout[-3000:], res.stderr[-3000:], sep="\n")
        sys.exit("mfa %s failed (exit %d)" % (args[0], res.returncode))
    return res.stdout + res.stderr
