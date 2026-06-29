"""CAIS-calibrated Tenable mock."""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import exp2_generate_mock as exp2  # noqa: E402
from exp2_common import DEFAULT_NVD_DIR, PROJECT_ROOT, log  # noqa: E402

STATS_FILE = PROJECT_ROOT / "cais_mock" / "real_stats.json"
CVES_FILE = PROJECT_ROOT / "cais_mock" / "cve_set.json"
DEFAULT_OUT = PROJECT_ROOT / "data" / "mock" / "cais_mock.json"

TRUTHY = {"true", "True"}
FALSY = {"false", "False"}

SOLUTION_TEMPLATES = (
    "n/a",
    "Apply vendor patches.",
    "See vendor advisory.",
    "Upgrade to patched version.",
    "Update to the latest version.",
    "Upgrade the affected package.",
    "Refer to vendor documentation.",
    "Install the latest security updates.",
    "Update affected software per vendor guidance.",
    "Upgrade the affected package; see advisory.",
)


def pick(rng: random.Random, dist: dict[str, int], exclude: set[str] | None = None):
    if not dist:
        return None
    items = [(k, v) for k, v in dist.items() if not (exclude and k in exclude)]
    if not items:
        return None
    keys, weights = zip(*items)
    return rng.choices(keys, weights=weights)[0]


def pick_int(rng: random.Random, dist: dict[str, int], default: int = 0) -> int:
    v = pick(rng, dist, exclude={"None", "Desconhecida", "None_"})
    try:
        return int(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def pick_bool(rng: random.Random, dist: dict[str, int]) -> bool:
    counts = {"True": 0, "False": 0}
    for k, v in dist.items():
        if k in TRUTHY:
            counts["True"] += v
        elif k in FALSY:
            counts["False"] += v
    return pick(rng, counts) == "True" if counts["True"] + counts["False"] else False


def sample_bin_uniform(rng: random.Random, bucket: str) -> float:
    lo, hi = bucket.strip("[)").split(",")
    return rng.uniform(float(lo), float(hi))


def pick_bin_float(rng: random.Random, dist: dict[str, int], present_rate: float) -> float | None:
    if rng.random() >= present_rate:
        return None
    b = pick(rng, dist)
    return round(sample_bin_uniform(rng, b), 2) if b else None


def coerce_severity(cvss3_score: float | None, sampler_value: str) -> int:
    if cvss3_score is not None:
        if cvss3_score < 4:
            return 1
        if cvss3_score < 7:
            return 2
        if cvss3_score < 9:
            return 3
        return 4
    try:
        return int(sampler_value)
    except (TypeError, ValueError):
        return 2


def fake_hostname(rng: random.Random, faker) -> str:
    env = rng.choices(["prod", "hml", "dev"], weights=[6, 3, 1])[0]
    return f"{faker.domain_word()}-{faker.word()}-{env}"


def sample_tag_list(rng: random.Random, faker, n: int, tag_cats: dict[str, int]) -> list[dict[str, str]]:
    return [
        {
            "id": str(uuid.UUID(int=rng.getrandbits(128))),
            "category": pick(rng, tag_cats) or faker.word().upper(),
            "value": faker.word().upper(),
            "type": rng.choices(["static", "dynamic"], weights=[7, 3])[0],
        }
        for _ in range(n)
    ]


def fabricate_cve_id(rng: random.Random, excluded: set[str]) -> str:
    for _ in range(16):
        cid = f"CVE-{rng.randint(1999, 2024)}-{rng.randint(1000, 99999)}"
        if cid not in excluded:
            return cid
    return f"CVE-{rng.randint(2025, 2099)}-{rng.randint(100000, 999999)}"


def build_record(cve: exp2.CVEFacts, stats: dict[str, Any], excluded_cves: set[str],
                 rng: random.Random, faker) -> dict[str, Any]:
    published_dt = exp2._date_to_dt(cve.published) if cve.published else datetime(2022, 1, 1)
    age_bucket = pick(rng, stats["age_hist"])
    age_days = int(sample_bin_uniform(rng, age_bucket)) if age_bucket else rng.randint(30, 730)
    last_seen_dt = datetime(2024, 7, 31) - timedelta(days=max(0, age_days - rng.randint(0, 30)))
    first_observed_dt = last_seen_dt - timedelta(days=rng.randint(1, max(1, age_days)))

    cvss3 = cve.cvss3 or {"base_score": None, "base_vector": None, "temporal_score": None, "temporal_vector": None}
    cvss2 = cve.cvss2 or {"base_score": None, "base_vector": None, "temporal_score": None, "temporal_vector": None}

    cvss4_score = pick_bin_float(rng, stats["def_cvss4_hist"], present_rate=0.065)
    cvss4 = {"base_score": cvss4_score, "base_vector": None, "temporal_score": None, "temporal_vector": None} if cvss4_score is not None else {}

    epss_score = pick_bin_float(rng, stats["def_epss_hist"], present_rate=0.0015)
    epss = {"score": epss_score} if epss_score is not None else {}

    severity = coerce_severity(cvss3.get("base_score"), pick(rng, stats["severity"]) or "2")

    os_head = pick(rng, stats["asset_os_head"]) or "Ubuntu"
    hostname = fake_hostname(rng, faker)

    n_tags = pick_int(rng, stats["asset_tag_count_hist"], default=3)
    n_ipv4 = max(1, pick_int(rng, stats["asset_ipv4_count_hist"], default=1))
    n_cve = pick_int(rng, stats["def_cve_count_hist"], default=1)
    n_cwe = pick_int(rng, stats["def_cwe_count_hist"], default=0)

    cve_field = None if n_cve == 0 else [cve.cve_id] + [fabricate_cve_id(rng, excluded_cves) for _ in range(n_cve - 1)]
    cwe_field = None if n_cwe == 0 else [pick(rng, stats["def_cwe_top"]) or "20" for _ in range(n_cwe)]

    definition: dict[str, Any] = {
        "id": rng.randint(10_000, 999_999),
        "name": exp2.synth_definition_name(cve),
        "description": cve.description,
        "solution": rng.choice(SOLUTION_TEMPLATES),
        "synopsis": "The remote service is affected by a vulnerability.",
        "see_also": [f"https://nvd.nist.gov/vuln/detail/{cve.cve_id}"],
        "family": pick(rng, stats["def_family"]) or "General",
        "severity": severity,
        "cpe": list(cve.cpe_uris[:max(1, n_cve)]),
        "cve": cve_field,
        "cwe": cwe_field,
        "exploitability_ease": pick(rng, stats["def_exploit_ease"]) or "NOT_AVAILABLE",
        "default_account": pick_bool(rng, stats["def_default_account"]),
        "exploited_by_malware": pick_bool(rng, stats["def_exploited_by_malware"]),
        "exploited_by_nessus": pick_bool(rng, stats["def_exploited_by_nessus"]),
        "unsupported_by_vendor": pick_bool(rng, stats["def_unsupported_by_vendor"]),
        "plugin_published": cve.published or exp2._isoformat(published_dt),
        "plugin_updated": cve.last_modified or exp2._isoformat(last_seen_dt),
        "vulnerability_published": cve.published or exp2._isoformat(published_dt),
        "patch_published": exp2._isoformat(published_dt + timedelta(days=rng.randint(0, 30))),
        "vpr": {"score": round(rng.uniform(0.1, 9.9), 1)},
        "vpr_v2": {},
        "cvss2": cvss2,
        "cvss3": cvss3,
        "cvss4": cvss4,
        "epss": epss,
    }

    asset: dict[str, Any] = {
        "id": str(uuid.UUID(int=rng.getrandbits(128))),
        "name": hostname,
        "tags": sample_tag_list(rng, faker, n_tags, stats["asset_tag_category"]),
        "criticality": str(pick_int(rng, stats["asset_criticality"], default=5)),
        "ipv4_addresses": [exp2.fake_ipv4(rng) for _ in range(n_ipv4)],
        "display_ipv4_address": exp2.fake_ipv4(rng),
        "display_ipv6_address": exp2.fake_ipv6(rng),
        "network": {"id": "00000000-0000-0000-0000-000000000000", "name": "Default"},
        "host_name": hostname,
        "operating_system": os_head,
        "system_type": pick(rng, stats["asset_system_type"]) or "general-purpose",
        "display_mac_address": exp2.fake_mac(rng),
    }
    if rng.random() < 0.4:
        asset["display_fqdn"] = f"{hostname}.{faker.domain_name()}"

    return {
        "output": exp2.synth_output(rng, faker),
        "id": str(uuid.UUID(int=rng.getrandbits(128))),
        "asset": asset,
        "definition": definition,
        "asset_cloud_resource": {"id": str(uuid.UUID(int=rng.getrandbits(128))),
                                  "name": f"{faker.domain_word()}-{faker.word()}-{rng.randint(1000,9999):04d}"},
        "container_image": {"id": str(uuid.UUID(int=rng.getrandbits(128))),
                             "name": f"{faker.domain_word()}-{faker.word()}:{rng.randint(1,9)}.{rng.randint(0,9)}.{rng.randint(0,9)}"},
        "severity": severity,
        "state": pick(rng, stats["state"]) or "ACTIVE",
        "first_observed": exp2._isoformat(first_observed_dt),
        "last_seen": exp2._isoformat(last_seen_dt),
        "risk_modified": exp2._isoformat(last_seen_dt),
        "protocol": pick(rng, stats["protocol"]) or "TCP",
        "port": pick_int(rng, stats["port_top"], default=0),
        "scan": {"id": str(uuid.UUID(int=rng.getrandbits(128))),
                  "target": asset["display_ipv4_address"]},
        "age_in_days": age_days,
        "vuln_age": age_days * 86_400_000,
    }


def stream_records(paths: Iterable[Path], excluded_cves: set[str], stats: dict[str, Any],
                   rng: random.Random, faker, limit: int | None) -> Iterator[dict[str, Any]]:
    n = 0
    for path in paths:
        if limit is not None and n >= limit:
            return
        facts = exp2.load_cve(path)
        if facts is None or facts.cve_id in excluded_cves:
            continue
        yield build_record(facts, stats, excluded_cves, rng, faker)
        n += 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-n", "--n-records", type=int, default=0,
                   help="0 = match unique_record_count from --stats")
    p.add_argument("--nvd-dir", type=Path, default=DEFAULT_NVD_DIR)
    p.add_argument("--stats", type=Path, default=STATS_FILE)
    p.add_argument("--cve-exclude", type=Path, default=CVES_FILE)
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--locale", default="pt_BR")
    p.add_argument("--year-min", type=int, default=1999)
    p.add_argument("--year-max", type=int, default=2024)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stats = json.loads(args.stats.read_text())
    excluded = set(json.loads(args.cve_exclude.read_text()))
    Faker = exp2.ensure_faker()
    faker = Faker(args.locale)
    Faker.seed(args.seed)
    rng = random.Random(args.seed)

    limit = args.n_records or stats.get("_meta", {}).get("total_record_count") or None
    log.info("CAIS-calibrated mock - excluding %d real CAIS CVEs; target=%s records",
             len(excluded), "all" if limit is None else limit)

    paths = list(exp2.iter_cve_files(args.nvd_dir, args.year_min, args.year_max))
    rng.shuffle(paths)

    count = exp2.stream_write_json_array(
        args.out_json,
        stream_records(iter(paths), excluded, stats, rng, faker, limit=limit),
    )
    if count == 0:
        log.error("nothing written - check --nvd-dir and --cve-exclude")
        return 2
    if limit is not None and count < limit:
        log.warning("usable NVD CVEs < requested after exclusion: %d < %d", count, limit)
    log.info("wrote %s (%d records)", args.out_json, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
