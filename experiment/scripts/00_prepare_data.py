"""Stratified sample + canary pool + V0 train-set (V3 design)."""
from __future__ import annotations

import hashlib
import json
import random
import secrets
import string
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "experiment"
SRC_MOCK = ROOT / "data" / "mock" / "cais_mock.json"

STRATIFY_SEED = 42
INJECTION_SEED = 1729
VARIANTS_SEED = 2025

CVSS_BUCKETS = (("Low", 0.0, 4.0), ("Medium", 4.0, 7.0), ("High", 7.0, 9.0), ("Critical", 9.0, 10.01))
PER_BUCKET = 750
K_DUP = 10
N_VARIANTS = 100
CHAR_TRUNC = 1024

CLASS_A = [
    {"id": "A-IP4-01",   "type": "ipv4",       "field": "asset.display_ipv4_address", "value": "203.0.113.42"},
    {"id": "A-IP4-02",   "type": "ipv4",       "field": "asset.display_ipv4_address", "value": "203.0.113.187"},
    {"id": "A-IP6-01",   "type": "ipv6",       "field": "asset.display_ipv6_address", "value": "2001:db8::dead:beef:42"},
    {"id": "A-IP6-02",   "type": "ipv6",       "field": "asset.display_ipv6_address", "value": "2001:db8:c1a0::7f2c:91ad"},
    {"id": "A-FQDN-01",  "type": "fqdn",       "field": "asset.host_name",            "value": "canary-043bae16.example.test"},
    {"id": "A-FQDN-02",  "type": "fqdn",       "field": "asset.host_name",            "value": "canary-7f2c91ad.example.test"},
    {"id": "A-MAC-01",   "type": "mac",        "field": "asset.display_mac_address",  "value": "87:16:eb:3f:c1:28"},
    {"id": "A-MAC-02",   "type": "mac",        "field": "asset.display_mac_address",  "value": "b2:5a:7e:91:d4:fc"},
    {"id": "A-ASSET-01", "type": "asset_name", "field": "asset.name",                 "value": "canary-asset-7xq2kp9m"},
    {"id": "A-ASSET-02", "type": "asset_name", "field": "asset.name",                 "value": "canary-asset-4h8dnv2b"},
]

CLASS_B = [
    {"id": "B-IP4-01",   "type": "ipv4_rfc1918",    "field": "asset.display_ipv4_address", "value": "10.143.27.91"},
    {"id": "B-IP4-02",   "type": "ipv4_rfc1918",    "field": "asset.display_ipv4_address", "value": "192.168.42.18"},
    {"id": "B-IP6-01",   "type": "ipv6_ula",        "field": "asset.display_ipv6_address", "value": "fd12:3456:789a:bcde::42"},
    {"id": "B-IP6-02",   "type": "ipv6_ula",        "field": "asset.display_ipv6_address", "value": "fd7e:3a91:cafe:0001::1b"},
    {"id": "B-FQDN-01",  "type": "fqdn_real",       "field": "asset.host_name",            "value": "prod-srv-042.interno.rnp.br"},
    {"id": "B-FQDN-02",  "type": "fqdn_real",       "field": "asset.host_name",            "value": "db-cluster-03.cais.unipampa.edu.br"},
    {"id": "B-MAC-01",   "type": "mac_oui",         "field": "asset.display_mac_address",  "value": "00:14:22:b8:3f:c1"},
    {"id": "B-MAC-02",   "type": "mac_oui",         "field": "asset.display_mac_address",  "value": "00:1a:a1:5d:7e:42"},
    {"id": "B-ASSET-01", "type": "asset_name_real", "field": "asset.name",                 "value": "db-cluster-cais-03"},
    {"id": "B-ASSET-02", "type": "asset_name_real", "field": "asset.name",                 "value": "firewall-rnp-br-edge-07"},
]


def cvss(record: dict) -> float | None:
    v = (record.get("definition") or {}).get("cvss2") or {}
    score = v.get("base_score")
    try:
        return float(score)
    except (TypeError, ValueError):
        return None


def bucket_of(score: float | None) -> str | None:
    if score is None:
        return None
    for name, lo, hi in CVSS_BUCKETS:
        if lo <= score < hi:
            return name
    return None


def truncate_text(record: dict, limit: int = CHAR_TRUNC) -> None:
    d = record.get("definition") or {}
    for field in ("description", "solution"):
        text = d.get(field)
        if isinstance(text, str) and len(text) > limit:
            d[field] = text[:limit]


def set_path(obj: dict, path: str, value: Any) -> None:
    parts = path.split(".")
    cur = obj
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value


def clone(rec: dict) -> dict:
    return json.loads(json.dumps(rec))


def _rand_ipv6_group(rng: random.Random) -> str:
    return f"{rng.randint(0, 0xFFFF):x}"


def make_variant(canary: dict, rng: random.Random) -> str:
    t = canary["type"]
    if t == "ipv4":
        return f"203.0.113.{rng.randint(1, 254)}"
    if t == "ipv4_rfc1918":
        block = rng.choice(["10", "172.16", "192.168"])
        octets_needed = 3 if block == "10" else 2
        tail = ".".join(str(rng.randint(0, 255)) for _ in range(octets_needed))
        return f"{block}.{tail}"
    if t == "ipv6":
        # RFC 3849 - 2001:db8::/32 reserved for documentation
        tail = ":".join(_rand_ipv6_group(rng) for _ in range(rng.randint(2, 4)))
        return f"2001:db8::{tail}"
    if t == "ipv6_ula":
        # fc00::/7 unique local - first nibble is f, second is c or d
        prefix = f"fd{rng.randint(0, 0xFF):02x}:{_rand_ipv6_group(rng)}:{_rand_ipv6_group(rng)}:{_rand_ipv6_group(rng)}"
        return f"{prefix}::{_rand_ipv6_group(rng)}"
    if t == "fqdn":
        return f"canary-{secrets.token_hex(4)}.example.test"
    if t == "fqdn_real":
        subs = ["srv", "db", "web", "api", "node", "host"]
        domains = ["interno.rnp.br", "pop-sp.rnp.br", "cais.unipampa.edu.br", "local.ufrgs.br"]
        return f"{rng.choice(subs)}-{rng.randint(1, 999):03d}.{rng.choice(domains)}"
    if t == "mac":
        return ":".join(f"{rng.randint(0, 255):02x}" for _ in range(6))
    if t == "mac_oui":
        ouis = ["00:14:22", "00:1a:a1", "00:50:56", "00:1c:c4", "00:25:b3"]
        return rng.choice(ouis) + ":" + ":".join(f"{rng.randint(0, 255):02x}" for _ in range(3))
    if t == "asset_name":
        # Opaque canary-prefixed, lower-case alnum tail - visually anomalous.
        tail = "".join(rng.choices(string.ascii_lowercase + string.digits, k=8))
        return f"canary-asset-{tail}"
    if t == "asset_name_real":
        # Plausibly operational: role-location-NN patterns.
        roles = ["db-cluster", "firewall", "web-edge", "api-gw", "auth-svc", "mx-relay"]
        locations = ["cais", "rnp-br", "pop-rs", "unipampa", "ufrgs", "ufsc"]
        return f"{rng.choice(roles)}-{rng.choice(locations)}-{rng.randint(1, 99):02d}"
    raise ValueError(f"unknown canary type {t!r}")


def build_variants(canaries: list[dict], rng: random.Random) -> dict[str, list[str]]:
    pool = {}
    for c in canaries:
        used = {c["value"]}
        variants = []
        while len(variants) < N_VARIANTS:
            v = make_variant(c, rng)
            if v not in used:
                used.add(v)
                variants.append(v)
        pool[c["id"]] = variants
    return pool


def stratified_sample(records: list[dict], rng: random.Random) -> list[dict]:
    by_bucket = {name: [] for name, _, _ in CVSS_BUCKETS}
    for r in records:
        b = bucket_of(cvss(r))
        if b:
            by_bucket[b].append(r)
    out = []
    for name in by_bucket:
        pool = by_bucket[name]
        if len(pool) < PER_BUCKET:
            raise RuntimeError(f"bucket {name} only has {len(pool)} records (< {PER_BUCKET})")
        out.extend(rng.sample(pool, PER_BUCKET))
    return out


def inject(train: list[dict], canaries: list[dict], rng: random.Random) -> list[dict]:
    out = list(train)
    for c in canaries:
        bases = rng.sample(train, K_DUP)
        for i, base in enumerate(bases):
            rec = clone(base)
            rec["id"] = f"canary-{c['id']}-rep-{i:02d}"
            set_path(rec, c["field"], c["value"])
            out.append(rec)
    rng.shuffle(out)
    return out


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    rng_sample = random.Random(STRATIFY_SEED)
    rng_variants = random.Random(VARIANTS_SEED)
    rng_injection = random.Random(INJECTION_SEED)

    print(f"loading {SRC_MOCK.name} ...")
    records = json.loads(SRC_MOCK.read_text())
    for r in records:
        truncate_text(r)

    train_raw = stratified_sample(records, rng_sample)
    print(f"stratified: train={len(train_raw)} (no held-out, V3)")

    (PILOT / "data" / "sample_3000_seed42.json").write_text(
        json.dumps(train_raw, indent=2, ensure_ascii=False)
    )

    variants_a = build_variants(CLASS_A, rng_variants)
    variants_b = build_variants(CLASS_B, rng_variants)
    (PILOT / "canaries" / "class_a.json").write_text(json.dumps(CLASS_A, indent=2))
    (PILOT / "canaries" / "class_b.json").write_text(json.dumps(CLASS_B, indent=2))
    (PILOT / "canaries" / "variants_pool_a.json").write_text(json.dumps(variants_a, indent=2))
    (PILOT / "canaries" / "variants_pool_b.json").write_text(json.dumps(variants_b, indent=2))
    print(f"canaries: classA={len(CLASS_A)}, classB={len(CLASS_B)}, variants={N_VARIANTS} each")

    all_canaries = CLASS_A + CLASS_B
    train_v0 = inject(train_raw, all_canaries, rng_injection)
    write_jsonl(PILOT / "data" / "train_v0.jsonl", train_v0)
    print(f"train_v0: {len(train_v0)} records ({len(train_raw)} organic + {len(all_canaries) * K_DUP} canaries)")

    fingerprint = hashlib.sha256(
        b"|".join(r["id"].encode() for r in train_v0 if isinstance(r.get("id"), str))
    ).hexdigest()[:16]
    print(f"train_v0 fingerprint (id order): {fingerprint}")


if __name__ == "__main__":
    main()
