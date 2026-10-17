"""Shared benchmark workloads of the fast runner: realistic SOC log windows with many yes/no questions and one tool-selection question.

    python -m sepal.runner.workloads [--out runs/runner-bench/workloads] [--only SHAPE ...]

Deterministic (every random choice is seeded by the shape name and the request index). Real log text from `runs/data` (read only):
Wazuh alerts of the AIT-ADS testbed (syslog, Apache, auditd lines), the LANL authentication log, Sysmon events of the Splunk attack data.
A request is one window of one family (a contiguous run of lines at a seeded offset) plus questions built from entities in the window
(present or absent) with custom criteria, as a Decision-Index `/v1/systemone` body. Each line of a workload file is

    {"id", "shape", "body", "input_ids", "question_types", "instruction_spans", "option_offsets", "option_spans",
     "question_ids", "option_ids", "tokens", "state_tokens", "family", "target_tokens", "floor"}

where the id lists and spans come from Unsloth's own encoder (`clef.encode_record`) on exactly what `sepal.serve.decision_api` hands it
(`to_record`, `conv.state_text`, `conv.request_questions`), so a C++ bench needs no Python encoder. Shapes `t{1k,4k,16k}_q{1,8,32}`
(`target_tokens` 1,024 / 4,096 / 16,000, so a request fits the oracle's 16,384-token window with the schema included: the state is cut to
what is left after the questions), `shared_t4k_q8` (8 requests, one 4k state, different questions: prefix reuse) and `t32k_q8` (runner
only: over the oracle's window). A shape whose questions alone exceed the target keeps a 256-token state and says so (`floor`: true).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

DATA = Path(r"E:\MYCOMPANY\BlumSOC\Sepal\runs\data")
OUT = Path(r"E:\MYCOMPANY\BlumSOC\Sepal\runs\runner-bench\workloads")
VERSION = 1
ORACLE_WINDOW = 16384
MIN_STATE_TOKENS = 256
TARGETS = {"1k": 1024, "4k": 4096, "16k": 16000, "32k": 32000}
COUNTS = {("1k", 1): 64, ("1k", 8): 64, ("1k", 32): 64, ("4k", 1): 32, ("4k", 8): 32, ("4k", 32): 32,
          ("16k", 1): 12, ("16k", 8): 12, ("16k", 32): 12}
SHARED_REQUESTS = 8
LONG_REQUESTS = 6
FAMILIES = ("wazuh", "auth", "sysmon")


# ---- the line pools (real text) -----------------------------------------------------------------------------------------------

def wazuh_pool(limit: int = 60000) -> list[dict]:
    out: list[dict] = []
    for name in ("fox", "harrison", "wheeler"):
        with open(DATA / "ait-ads" / "extracted" / f"{name}_wazuh.json", encoding="utf-8") as f:
            for n, line in enumerate(f):
                if n >= limit:
                    break
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rule, pre = r.get("rule", {}), r.get("predecoder", {})
                host, prog = str(pre.get("hostname", "-")), str(pre.get("program_name", "-"))
                text = (f"{str(r.get('@timestamp', ''))[:19]}Z {host} {prog}[rule {rule.get('id')} L{rule.get('level')}] "
                        f"{rule.get('description', '')} :: {str(r.get('full_log', '')).strip()}")
                out.append({"line": " ".join(text.split()) if "\n" in text else text, "rule": str(rule.get("id")), "level": int(rule.get("level", 0)),
                            "desc": str(rule.get("description", "")), "host": host, "prog": prog, "src": name})
    return out


def auth_pool(limit: int = 400000) -> list[dict]:
    out: list[dict] = []
    with gzip.open(DATA / "lanl-cyber1" / "auth.txt.gz", "rt", encoding="utf-8") as f:
        for n, line in enumerate(f):
            if n >= limit:
                break
            p = line.strip().split(",")
            if len(p) != 9:
                continue
            out.append({"line": line.strip(), "user": p[1], "dst_user": p[2], "src": p[3], "dst": p[4], "auth": p[5], "logon": p[6],
                        "orient": p[7], "outcome": p[8]})
    return out


def sysmon_pool(per_file: int = 4000) -> list[dict]:
    root = DATA / "splunk-attack-data" / "datasets" / "attack_techniques"
    files = sorted(p for p in root.rglob("windows-sysmon.log") if p.stat().st_size > 300_000)
    out: list[dict] = []
    for path in files:
        with open(path, encoding="utf-8", errors="replace") as f:
            for n, line in enumerate(f):
                if n >= per_file:
                    break
                eid = re.search(r"<EventID>(\d+)</EventID>", line)
                if not eid:
                    continue
                ts = (re.search(r"SystemTime='([^']+)'", line) or [None, ""])[1][:19]
                comp = (re.search(r"<Computer>([^<]*)<", line) or [None, "-"])[1]
                data = [(k, v) for k, v in re.findall(r"<Data Name='([^']+)'>([^<]*)</Data>", line) if k not in ("RuleName", "ProcessGuid", "ParentProcessGuid", "LogonGuid", "UtcTime")]
                text = f"{ts}Z {comp} sysmon EID={eid.group(1)} " + " ".join(f"{k}={v}" for k, v in data)
                fields = dict(data)
                out.append({"line": text, "eid": eid.group(1), "host": comp, "image": fields.get("Image", ""), "cmd": fields.get("CommandLine", ""),
                            "user": fields.get("User", ""), "port": fields.get("DestinationPort", ""), "ip": fields.get("DestinationIp", ""),
                            "file": fields.get("TargetFilename", ""), "src": path.parent.parent.name + "/" + path.parent.name})
    return out


def load_pools(cache: Path) -> dict[str, list[dict]]:
    """The three pools, cached as JSONL under `cache` (extraction from the raw data is deterministic and takes a minute)."""
    cache.mkdir(parents=True, exist_ok=True)
    makers = {"wazuh": wazuh_pool, "auth": auth_pool, "sysmon": sysmon_pool}
    pools = {}
    for name, make in makers.items():
        path = cache / f"{name}.jsonl"
        if not path.is_file():
            tmp = path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for row in make():
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            os.replace(tmp, path)
        with open(path, encoding="utf-8") as f:
            pools[name] = [json.loads(line) for line in f]
    return pools


# ---- questions -------------------------------------------------------------------------------------------------------------

TOOLS = {"lookup_ip_reputation": "Look an address up in the threat intelligence feeds",
         "query_siem": "Search the SIEM for related events", "isolate_host": "Cut a host off from the network",
         "reset_credentials": "Force a password reset for an account", "open_ticket": "Open a ticket for a human analyst",
         "block_ip_firewall": "Add an address to the perimeter deny list", "get_process_tree": "Fetch the process tree of a host",
         "check_file_hash": "Check a file hash against malware databases"}


def _pick(rng: random.Random, items: list, default=None):
    return rng.choice(items) if items else default


def _entity(rng: random.Random, present: list, absent: list):
    """An entity of the window (60 percent of the time) or one that is not in it."""
    use_present = rng.random() < 0.6 and bool(present)
    return rng.choice(present if use_present else (absent or present))


def entities(family: str, rows: list[dict], pool: list[dict], rng: random.Random) -> dict:
    """Entities present in the window, and a few absent ones (seen elsewhere in the family's pool), per kind."""
    keys = {"wazuh": ("rule", "host", "prog"), "auth": ("user", "src", "dst", "auth"), "sysmon": ("eid", "host", "image", "port")}[family]
    present = {k: sorted({r[k] for r in rows if r.get(k) not in ("", "-", "?", None)}) for k in keys}
    everything = {k: sorted({r[k] for r in rng.sample(pool, min(len(pool), 3000)) if r.get(k) not in ("", "-", "?", None)}) for k in keys}
    absent = {k: [v for v in everything[k] if v not in set(present[k])] for k in keys}
    ips = sorted({m for r in rows for m in re.findall(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", r["line"])})
    return {"present": present, "absent": absent, "ips": ips}


def build_question_texts(family: str, rows: list[dict], ents: dict, rng: random.Random, n: int, terse: bool) -> list[tuple[str, str, str]]:
    """n distinct (instruction, true description, false description) triples about the window."""
    present, absent = ents["present"], ents["absent"]
    pending: list[tuple[str, str, str]] = []

    def add(instruction: str, yes: str, no: str) -> None:
        if terse:
            pending.append((instruction, yes.split(";")[0], no.split(";")[0]))
        else:
            pending.append((instruction, yes, no))

    levels = sorted({r["level"] for r in rows}) if family == "wazuh" else []
    for _ in range(n * 4 + 40):
        if family == "wazuh":
            kind = rng.randrange(7)
            if kind == 0 and (present["rule"] or absent["rule"]):
                rule = _entity(rng, present["rule"], absent["rule"])
                add(f"Does the window contain an alert of rule {rule}?", f"At least one alert has rule id {rule}; the rule fired in the window",
                    f"No alert has rule id {rule}; the rule did not fire in the window")
            elif kind == 1:
                k = rng.choice([3, 5, 7, 9, 10])
                add(f"Is there an alert of level {k} or higher?", f"At least one alert is of level {k} or above; escalate it",
                    f"Every alert is below level {k}; nothing to escalate")
            elif kind == 2 and (present["prog"] or absent["prog"]):
                prog = _entity(rng, present["prog"], absent["prog"])
                add(f"Does the program {prog} log anything in the window?", f"The program {prog} appears in at least one line",
                    f"The program {prog} does not appear in any line")
            elif kind == 3 and (present["host"] or absent["host"]):
                host = _entity(rng, present["host"], absent["host"])
                add(f"Is host {host} reporting in this window?", f"At least one line comes from host {host}", f"No line comes from host {host}")
            elif kind == 4 and ents["ips"]:
                ip = rng.choice(ents["ips"])
                add(f"Is the address {ip} involved in any alert?", f"The address {ip} appears in at least one line", f"The address {ip} appears in no line")
            elif kind == 5:
                add("Is there a failed authentication in the window?", "A line reports a failed login or a failed password; treat it as a failed authentication",
                    "No line reports a failed login or a failed password")
            else:
                word = rng.choice(["error", "denied", "root", "sudo", "invalid", "session opened", "segfault", "POST"])
                add(f"Does any line contain the word {word}?", f"The word {word} appears in at least one line", f"The word {word} appears in no line")
        elif family == "auth":
            kind = rng.randrange(6)
            if kind == 0 and (present["user"] or absent["user"]):
                user = _entity(rng, present["user"], absent["user"])
                add(f"Does the account {user} authenticate in this window?", f"The account {user} appears as the source user of at least one event",
                    f"The account {user} does not appear as a source user")
            elif kind == 1 and (present["dst"] or absent["dst"]):
                comp = _entity(rng, present["dst"], absent["dst"])
                add(f"Is computer {comp} the target of an authentication?", f"At least one event authenticates to {comp}", f"No event authenticates to {comp}")
            elif kind == 2:
                add("Is there a failed authentication?", "At least one event ends with Fail; a logon failed", "Every event ends with Success")
            elif kind == 3 and (present["auth"] or absent["auth"]):
                a = rng.choice(["NTLM", "Kerberos", "Negotiate", "MICROSOFT_AUTHENTICATION_PACKAGE_V1_0"])
                add(f"Does any event use the {a} package?", f"At least one event uses {a}", f"No event uses {a}")
            elif kind == 4 and (present["src"] or absent["src"]):
                comp = _entity(rng, present["src"], absent["src"])
                add(f"Do logons start from computer {comp}?", f"At least one event has {comp} as its source computer", f"No event has {comp} as its source computer")
            else:
                add("Does one account log on to many different computers?", "One account reaches five or more different computers; possible lateral movement",
                    "No account reaches more than a few computers")
        else:
            kind = rng.randrange(7)
            if kind == 0 and (present["eid"] or absent["eid"]):
                eid = rng.choice(["1", "3", "7", "10", "11", "13", "22", "23"])
                add(f"Is Sysmon event {eid} logged in the window?", f"At least one event has EventID {eid}", f"No event has EventID {eid}")
            elif kind == 1 and (present["image"] or absent["image"]):
                image = _entity(rng, present["image"], absent["image"])
                name = image.replace("\\", "/").rsplit("/", 1)[-1]
                add(f"Does the process {name} appear in the window?", f"At least one event has the image {name}", f"No event has the image {name}")
            elif kind == 2:
                add("Is PowerShell executed in the window?", "A process image or command line is powershell or pwsh", "No event runs powershell or pwsh")
            elif kind == 3 and (present["host"] or absent["host"]):
                host = _entity(rng, present["host"], absent["host"])
                add(f"Do events come from computer {host}?", f"At least one event is logged by {host}", f"No event is logged by {host}")
            elif kind == 4:
                port = rng.choice(["22", "80", "135", "389", "443", "445", "3389", "5985"])
                add(f"Is there an outbound connection to port {port}?", f"A network event has destination port {port}", f"No network event has destination port {port}")
            elif kind == 5:
                add("Is a file written under a temporary directory?", "A file-creation event has a path under Temp or AppData", "No file-creation event writes under Temp or AppData")
            else:
                add("Does any command line use an encoded argument?", "A command line carries -enc, -EncodedCommand or a base64 blob", "No command line carries an encoded argument")
    seen, out = set(), []
    for item in pending:
        if item[0] not in seen:
            seen.add(item[0])
            out.append(item)
    rng.shuffle(out)
    if len(out) < n:
        raise RuntimeError(f"only {len(out)} distinct questions for a {family} window; need {n}")
    return out[:n]


def questions_body(texts: list[tuple[str, str, str]], rng: random.Random, with_tool: bool) -> dict:
    questions: dict = {}
    for i, (instruction, yes, no) in enumerate(texts):
        questions[f"q{i + 1:02d}"] = {"type": "noul", "instructions": instruction, "criteria": {"true": yes, "false": no}}
    if with_tool:
        names = list(TOOLS)
        rng.shuffle(names)                    # the options are sent in no particular order; the encoder sorts them
        questions["tool"] = {"type": "choice", "instructions": "Which tool should the analyst use next on this window?",
                             "criteria": {name: TOOLS[name] for name in names}}
    return questions


# ---- requests --------------------------------------------------------------------------------------------------------------

class Builder:
    def __init__(self, pools: dict[str, list[dict]], limit: int = ORACLE_WINDOW):
        from transformers import AutoTokenizer

        from sepal.runner import oracle

        self.clef = oracle.load_vendored_clef()
        self.tokenizer = AutoTokenizer.from_pretrained(str(oracle.FINAL), local_files_only=True)
        self.pools, self.limit = pools, limit

    def encode(self, body: dict, limit: int):
        """What decision_api hands the encoder for this body, encoded (one token over the limit is allowed to detect an overflow)."""
        from sepal.serve.decision_api import to_record
        from sepal.train import unsloth_convert as conv

        record = to_record(body)
        state, questions = conv.state_text(record["state"]), conv.request_questions(record)
        return self.clef.encode_record(self.tokenizer, {"state": state, "questions": questions}, max_length=limit + 1), state

    def window(self, family: str, start: int, count: int) -> list[dict]:
        pool = self.pools[family]
        return [pool[(start + i) % len(pool)] for i in range(count)]

    def fit_state(self, family: str, start: int, questions: dict, target: int, limit: int) -> tuple[str, list[dict], object, bool]:
        """The log window of `family` from `start` whose request has `target` tokens (state cut to what the questions leave), the lines, its
        encoding, and whether the 256-token floor was hit."""
        body = {"model": "sepal", "state": "", "questions": questions}
        base, _ = self.encode(body, limit)
        budget = max(MIN_STATE_TOKENS, target - len(base.input_ids))
        floor = target - len(base.input_ids) < MIN_STATE_TOKENS
        pool = self.pools[family]
        # token cost per line, then the longest run from `start` within the budget; two corrections against the real encoding
        lines = self.window(family, start, min(len(pool), budget // 6 + 50))
        costs = [len(ids) + 1 for ids in self.tokenizer([r["line"] for r in lines], add_special_tokens=False).input_ids]
        total, count = 0, 0
        for c in costs:
            if total + c > budget:
                break
            total, count = total + c, count + 1
        count = max(1, count)
        while True:                     # the estimate ignores token merges at line joins: step down until the real encoding fits
            text = "\n".join(r["line"] for r in lines[:count])
            enc, _ = self.encode({"model": "sepal", "state": text, "questions": questions}, limit)
            if len(enc.input_ids) <= len(base.input_ids) + budget or count <= 1:
                break
            count -= 1
        return text, lines[:count], enc, floor

    def request(self, shape: str, index: int, family: str, start: int, qcount: int, target: int, limit: int, question_seed: str | None = None,
                state_from: tuple | None = None) -> dict:
        rng = random.Random(question_seed or f"{shape}:{index}:q")
        terse = qcount >= 32
        with_tool = qcount > 1 or index % 2 == 1
        n_noul = qcount - (1 if with_tool and qcount > 1 else 0) if qcount > 1 else (0 if with_tool else 1)
        # entities from a generous window around the start (the final window is a prefix of it)
        probe = self.window(family, start, min(600, max(24, target // 30)))
        ents = entities(family, probe, self.pools[family], rng)
        texts = build_question_texts(family, probe, ents, rng, n_noul, terse)
        questions = questions_body(texts, rng, with_tool)
        if state_from is not None:
            text, lines, enc, floor = state_from
            body = {"model": "sepal", "state": text, "questions": questions}
            enc, _ = self.encode(body, limit)
        else:
            text, lines, enc, floor = self.fit_state(family, start, questions, target, limit)
            body = {"model": "sepal", "state": text, "questions": questions}
        state_ids = len(self.tokenizer(text, add_special_tokens=False).input_ids)
        qids = [q.question_id for q in enc.questions]
        return {"id": f"{shape}-{index:03d}", "shape": shape, "body": body, "input_ids": list(enc.input_ids),
                "question_types": [q.question_type for q in enc.questions], "instruction_spans": [list(q.question_span) for q in enc.questions],
                "option_offsets": _offsets(enc), "option_spans": [list(s) for q in enc.questions for s in q.option_spans],
                "question_ids": qids, "option_ids": [list(q.option_ids) for q in enc.questions], "tokens": len(enc.input_ids),
                "state_tokens": state_ids, "family": family, "target_tokens": target, "floor": floor,
                "_window": (text, lines, enc, floor)}


def _offsets(enc) -> list[int]:
    out = [0]
    for q in enc.questions:
        out.append(out[-1] + len(q.option_spans))
    return out


def shape_names() -> list[str]:
    names = [f"t{t}_q{q}" for t in ("1k", "4k", "16k") for q in (1, 8, 32)]
    return names + ["shared_t4k_q8", "t32k_q8"]


def make_shape(builder: Builder, shape: str) -> list[dict]:
    pools = builder.pools
    rows: list[dict] = []
    seen: set[str] = set()

    def starts(index: int, family: str) -> int:
        rng = random.Random(f"{shape}:{index}:start")
        return rng.randrange(len(pools[family]))

    if shape == "shared_t4k_q8":
        family = "wazuh"
        first = builder.request(shape, 0, family, starts(0, family), 8, TARGETS["4k"], ORACLE_WINDOW)
        window = first["_window"]
        rows.append(first)
        for i in range(1, SHARED_REQUESTS):
            rows.append(builder.request(shape, i, family, starts(0, family), 8, TARGETS["4k"], ORACLE_WINDOW, question_seed=f"{shape}:{i}:q", state_from=window))
        return rows
    long_shape = shape == "t32k_q8"
    t, q = (("32k", 8) if long_shape else re.fullmatch(r"t(\d+k)_q(\d+)", shape).groups())
    q = int(q)
    count = LONG_REQUESTS if long_shape else COUNTS[(t, q)]
    limit = 40000 if long_shape else ORACLE_WINDOW
    for i in range(count):
        family = FAMILIES[i % 3]
        for attempt in range(20):             # distinct states
            row = builder.request(shape, i, family, starts(i + attempt * 1000, family), q, TARGETS[t], limit)
            digest = hashlib.sha256(row["body"]["state"].encode("utf-8")).hexdigest()
            if digest not in seen:
                seen.add(digest)
                break
        rows.append(row)
    return rows


def write_shape(path: Path, rows: list[dict]) -> str:
    tmp = path.with_suffix(".tmp")
    h = hashlib.sha256()
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            line = json.dumps({k: v for k, v in row.items() if not k.startswith("_")}, ensure_ascii=False, separators=(",", ":")) + "\n"
            h.update(line.encode("utf-8"))
            f.write(line)
    os.replace(tmp, path)
    return h.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--only", nargs="*", help="shape names (default: all)")
    args = parser.parse_args(argv)
    from sepal.runner import oracle

    oracle.hide_gpu()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pools = load_pools(out / "_pools")
    builder = Builder(pools)
    manifest = {"version": VERSION, "oracle_window": ORACLE_WINDOW, "shapes": {}}
    manifest_path = out / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for shape in args.only or shape_names():
        rows = make_shape(builder, shape)
        digest = write_shape(out / f"{shape}.jsonl", rows)
        toks = [r["tokens"] for r in rows]
        manifest["shapes"][shape] = {"file": f"{shape}.jsonl", "requests": len(rows), "tokens_min": min(toks), "tokens_mean": round(sum(toks) / len(toks), 1),
                                     "tokens_max": max(toks), "state_tokens_mean": round(sum(r["state_tokens"] for r in rows) / len(rows), 1),
                                     "questions": len(rows[0]["question_types"]), "floor": any(r["floor"] for r in rows),
                                     "oracle_comparable": max(toks) <= ORACLE_WINDOW, "sha256": digest,
                                     "families": {f: sum(1 for r in rows if r["family"] == f) for f in FAMILIES}}
        print(shape, manifest["shapes"][shape], flush=True)
    tmp = manifest_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
