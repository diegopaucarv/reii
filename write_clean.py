import glob
import json
import os
import re

folder = r"data/txt_outputs/OTRO"
out_folder = r"data/txt_outputs/OTRO_clean"
os.makedirs(out_folder, exist_ok=True)

# Interviewer markers (same as identify_speakers.py)
interviewer_markers = [
    r"mi nombre es",
    r"formo parte",
    r"equipo (de|acad)",
    r"consentimiento",
    r"consentimiento informado",
    r"acepta",
    r"grabaci[oó]n",
    r"participaci[oó]n",
    r"voluntaria",
    r"no es una auditor",
    r"no venimos",
    r"universidad",
    r"investigaci[oó]n",
    r"quisiera preguntar",
    r"preguntar",
    r"le voy a (hacer|preguntar)",
    r"voy a (hacer|preguntar)",
    r"c[oó]mo lleg[oó]",
    r"cu[aá]nto tiempo",
    r"qu[eé] profesi[oó]n",
    r"qu[eé] cargo",
    r"c[oó]mo asumi[oó]",
    r"cu[aá]l es su",
    r"cu[aá]l fue",
    r"qu[eé] sinti[oó]",
    r"qu[eé] fue lo primero",
    r"qui[eé]nes estaban",
    r"usted lleg[oó]",
    r"usted tiene",
    r"usted ha",
    r"usted es",
    r"usted vive",
    r"usted reside",
    r"usted trabaja",
    r"usted particip",
    r"usted estuvo",
    r"usted me",
    r"usted se",
    r"usted con",
    r"usted en",
    r"usted para",
    r"usted de",
    r"usted y",
    r"usted a",
    r"usted no",
    r"usted qu",
    r"usted c",
    r"usted p",
    r"usted t",
    r"usted s",
    r"usted d",
    r"usted l",
    r"usted r",
    r"usted b",
    r"usted f",
    r"usted g",
    r"usted h",
    r"usted j",
    r"usted m",
    r"usted n",
    r"usted v",
    r"usted o",
    r"usted u",
    r"usted i",
    r"usted z",
    r"usted w",
    r"usted x",
    r"usted k",
    r"usted y",
]


def classify_speaker(texts):
    joined = " ".join(texts).lower()
    score = 0
    for marker in interviewer_markers:
        if re.search(marker, joined):
            score += 1
    if re.search(r"mi nombre es", joined):
        score += 3
    return score


manifest = []
for path in sorted(glob.glob(os.path.join(folder, "*.srt"))):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()
    blocks = re.split(r"\n\s*\n", content)
    speakers = {}
    order = []
    for block in blocks:
        lines = block.strip().split("\n")
        for l in lines:
            l = l.strip()
            if not l or re.match(r"^\d+$", l) or "-->" in l or l.startswith("["):
                continue
            if "EFECTO" in l or "MÚSICA" in l or "MUSICA" in l:
                continue
            m = re.match(r"SPEAKER_(\d+|UNKNOWN):\s*(.*)", l)
            if m:
                spk = m.group(1)
                text = m.group(2).strip()
                if text:
                    speakers.setdefault(spk, []).append(text)
                    if spk not in order:
                        order.append(spk)

    # Determine interviewer
    scores = {spk: classify_speaker(texts) for spk, texts in speakers.items()}
    interviewer = max(scores.items(), key=lambda x: x[1])[0] if scores else None

    # Build clean text with speaker labels
    clean_lines = []
    for spk in order:
        role = "INTERVIEWER" if spk == interviewer else "INTERVIEWEE"
        if spk == "UNKNOWN":
            role = "UNKNOWN"
        clean_lines.append(f"=== {role} (SPEAKER_{spk}) ===")
        for t in speakers[spk]:
            clean_lines.append(t)
        clean_lines.append("")

    clean_text = "\n".join(clean_lines)

    # Write clean file
    base = os.path.basename(path)
    safe_name = re.sub(r"[^\w\-\. ]", "_", base)
    out_path = os.path.join(out_folder, safe_name.replace(".srt", ".txt"))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(clean_text)

    all_texts = [t for texts in speakers.values() for t in texts]
    manifest.append(
        {
            "original": base,
            "clean": safe_name.replace(".srt", ".txt"),
            "interviewer": interviewer,
            "speakers": {k: len(v) for k, v in speakers.items()},
            "token_est": int(
                max(
                    sum(len(t.split()) for t in all_texts) * 1.3,
                    sum(len(t) for t in all_texts) / 4,
                )
            ),
        }
    )

with open("clean_manifest.json", "w", encoding="utf-8") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)

print(f"Wrote {len(manifest)} clean files to {out_folder}")
for m in manifest:
    print(f"{m['clean'][:50]:<50} INT={m['interviewer']} tok={m['token_est']}")
