"""Delete demo patients from the deployed Sleep Disorder dashboard.

The script uses the existing dashboard delete route, including Django CSRF, so
it works without adding a dedicated destructive API endpoint.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import quote

import requests


DEFAULT_BASE_URL = "http://sleep-portal-alb-67325866.ap-southeast-1.elb.amazonaws.com"
VALID_DIAGNOSES = {"healthy", "insomnia", "narcolepsy", "nfle", "plm", "rbd", "sdb"}


@dataclass(frozen=True)
class PatientRow:
    patient_id: str
    diagnosis: str


def _clean(value: str) -> str:
    return html.unescape(value).strip()


def fetch_patient_rows(session: requests.Session, base_url: str, timeout: int) -> list[PatientRow]:
    response = session.get(f"{base_url}/patients/", timeout=timeout)
    response.raise_for_status()

    rows: list[PatientRow] = []
    for row_html in re.findall(r"<tr\b.*?</tr>", response.text, flags=re.IGNORECASE | re.DOTALL):
        id_match = re.search(
            r'class="patient-code-sub"[^>]*>[^:<]*:\s*([^<]+)</div>',
            row_html,
            flags=re.IGNORECASE,
        )
        if not id_match:
            continue

        chips = [
            _clean(match)
            for match in re.findall(
                r'<span[^>]*class="[^"]*chip[^"]*chip-soft[^"]*"[^>]*>(.*?)</span>',
                row_html,
                flags=re.IGNORECASE | re.DOTALL,
            )
        ]
        diagnosis = chips[1] if len(chips) >= 2 else ""
        rows.append(PatientRow(patient_id=_clean(id_match.group(1)), diagnosis=diagnosis))
    return rows


def get_csrf_token(session: requests.Session, base_url: str, timeout: int) -> str:
    response = session.get(f"{base_url}/patients/", timeout=timeout)
    response.raise_for_status()
    match = re.search(r'name="csrfmiddlewaretoken"\s+value="([^"]+)"', response.text)
    if match:
        return match.group(1)
    token = session.cookies.get("csrftoken")
    if token:
        return token
    raise RuntimeError("Could not find CSRF token from /patients/.")


def select_patients(
    rows: Iterable[PatientRow],
    *,
    patient_ids: set[str],
    prefixes: list[str],
    diagnoses: set[str],
    unknown_diagnosis: bool,
    mixed_demo: bool,
    demo_rich: bool,
    realtime_demo: bool,
    quick_demo: bool,
    all_demo: bool,
) -> list[PatientRow]:
    selected: dict[str, PatientRow] = {}
    lowered_prefixes = [prefix.lower() for prefix in prefixes]
    lowered_diagnoses = {diagnosis.lower() for diagnosis in diagnoses}

    for row in rows:
        pid_lower = row.patient_id.lower()
        diagnosis_lower = row.diagnosis.lower()
        matched = False

        if row.patient_id in patient_ids:
            matched = True
        if lowered_prefixes and any(pid_lower.startswith(prefix) for prefix in lowered_prefixes):
            matched = True
        if lowered_diagnoses and diagnosis_lower in lowered_diagnoses:
            matched = True
        if unknown_diagnosis and diagnosis_lower not in VALID_DIAGNOSES:
            matched = True
        if mixed_demo and "mixed" in pid_lower:
            matched = True
        if demo_rich and pid_lower.startswith("demo-rich-"):
            matched = True
        if realtime_demo and pid_lower.startswith("iot-"):
            matched = True
        if quick_demo and pid_lower.startswith("demo-iot-"):
            matched = True
        if all_demo and (
            pid_lower.startswith("demo-rich-")
            or pid_lower.startswith("demo-iot-")
            or pid_lower.startswith("iot-")
        ):
            matched = True

        if matched:
            selected[row.patient_id] = row

    return sorted(selected.values(), key=lambda item: item.patient_id)


def delete_patient(
    session: requests.Session,
    *,
    base_url: str,
    patient_id: str,
    csrf_token: str,
    timeout: int,
) -> tuple[bool, int, str]:
    encoded_id = quote(patient_id, safe="")
    url = f"{base_url}/patients/{encoded_id}/delete/"
    response = session.post(
        url,
        data={"csrfmiddlewaretoken": csrf_token},
        headers={
            "X-CSRFToken": csrf_token,
            "Referer": f"{base_url}/patients/",
        },
        timeout=timeout,
        allow_redirects=False,
    )
    ok = response.status_code in {200, 302, 303}
    return ok, response.status_code, response.text[:160].replace("\n", " ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Delete demo patients through the dashboard.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--patient-id", action="append", default=[], help="Exact patient_id to delete.")
    parser.add_argument("--id-prefix", action="append", default=[], help="Delete patient_id values with this prefix.")
    parser.add_argument("--diagnosis", action="append", default=[], help="Delete patients with this diagnosis.")
    parser.add_argument("--unknown-diagnosis", action="store_true", help="Delete diagnoses outside the 7 official labels.")
    parser.add_argument("--mixed-demo", action="store_true", help="Delete demo patients whose patient_id contains 'mixed'.")
    parser.add_argument("--demo-rich", action="store_true", help="Delete rich demo patients whose patient_id starts with demo-rich-.")
    parser.add_argument("--realtime-demo", action="store_true", help="Delete realtime IoT demo patients whose patient_id starts with iot-.")
    parser.add_argument("--quick-demo", action="store_true", help="Delete quick demo patients whose patient_id starts with demo-iot-.")
    parser.add_argument("--all-demo", action="store_true", help="Delete demo-rich-, demo-iot-, and iot-* patients.")
    parser.add_argument("--list", action="store_true", help="Only list matching patients.")
    parser.add_argument("--yes", action="store_true", help="Actually delete. Without this flag the script is dry-run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_url = args.base_url.rstrip("/")

    with requests.Session() as session:
        rows = fetch_patient_rows(session, base_url, args.timeout)
        selected = select_patients(
            rows,
            patient_ids=set(args.patient_id),
            prefixes=args.id_prefix,
            diagnoses=set(args.diagnosis),
            unknown_diagnosis=args.unknown_diagnosis,
            mixed_demo=args.mixed_demo,
            demo_rich=args.demo_rich,
            realtime_demo=args.realtime_demo,
            quick_demo=args.quick_demo,
            all_demo=args.all_demo,
        )

        print(f"Found {len(rows)} patients on {base_url}/patients/")
        if not selected:
            print("No matching patients.")
            return

        action = "LIST" if args.list else ("DELETE" if args.yes else "DRY-RUN")
        print(f"{action}: {len(selected)} matching patients")
        for row in selected:
            print(f"  - {row.patient_id} | diagnosis={row.diagnosis}")

        if args.list or not args.yes:
            print("")
            print("Nothing was deleted. Add --yes to delete the listed patients.")
            return

        csrf_token = get_csrf_token(session, base_url, args.timeout)
        deleted = 0
        failed = 0
        for row in selected:
            ok, status_code, body_preview = delete_patient(
                session,
                base_url=base_url,
                patient_id=row.patient_id,
                csrf_token=csrf_token,
                timeout=args.timeout,
            )
            if ok:
                deleted += 1
                print(f"DELETED {row.patient_id} ({status_code})")
            else:
                failed += 1
                print(f"FAILED  {row.patient_id} ({status_code}) {body_preview}")

        print("")
        print(f"Done. deleted={deleted}, failed={failed}")


if __name__ == "__main__":
    main()
