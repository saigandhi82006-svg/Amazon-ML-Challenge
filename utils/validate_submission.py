"""
Official Submission Validator — Business Entity Resolution Challenge 2026
Verifies format, headers, delimiters, row counts, entity ID constraints, and candidate containment.
"""

import argparse
import sys
from pathlib import Path
import pandas as pd


def validate_submission(
    matching_results_path: Path,
    candidate_pairs_path: Path,
    test_source1_path: Path,
) -> bool:
    """
    Validates submission files against all competition requirements.
    """
    print("=" * 60)
    print("OFFICIAL SUBMISSION VALIDATOR — ML CHALLENGE 2026")
    print("=" * 60)

    errors = []

    # Check 1: File existence
    if not matching_results_path.exists():
        errors.append(f"Matching results file not found: {matching_results_path}")
    if not candidate_pairs_path.exists():
        errors.append(f"Candidate pairs file not found: {candidate_pairs_path}")
    if not test_source1_path.exists():
        errors.append(f"Reference Test Source 1 file not found: {test_source1_path}")

    if errors:
        for err in errors:
            print(f"[FAIL] {err}")
        return False

    # Check 2: Load Test Source 1 reference IDs
    print("\n1. Verifying Reference Source 1 Entity IDs...")
    try:
        test_s1_df = pd.read_csv(
            test_source1_path,
            sep="," if str(test_source1_path).endswith(".csv") else "\t",
            usecols=["entity_id"],
        )
        expected_s1_ids = list(test_s1_df["entity_id"])
        expected_s1_set = set(expected_s1_ids)
        print(f"  Reference Source 1 entities: {len(expected_s1_ids):,}")
    except Exception as e:
        print(f"[FAIL] Could not read reference Test Source 1 file: {e}")
        return False

    # Check 3: Validate candidate_pairs.tsv
    print("\n2. Validating candidate_pairs.tsv...")
    try:
        with open(candidate_pairs_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            if header != "source1_entity_id\tcandidate_entity_ids":
                errors.append(
                    f"candidate_pairs.tsv header mismatch. Expected: 'source1_entity_id\\tcandidate_entity_ids', Found: '{header}'"
                )
    except Exception as e:
        errors.append(f"Could not read candidate_pairs.tsv: {e}")

    # Read candidate pairs
    cand_dict = {}
    try:
        cand_df = pd.read_csv(candidate_pairs_path, sep="\t", dtype=str, keep_default_na=False)
        cand_s1_ids = list(cand_df["source1_entity_id"])
        if len(cand_s1_ids) != len(expected_s1_ids):
            errors.append(
                f"candidate_pairs.tsv row count ({len(cand_s1_ids):,}) != expected ({len(expected_s1_ids):,})"
            )
        if set(cand_s1_ids) != expected_s1_set:
            errors.append("candidate_pairs.tsv S1 entity IDs do not match Test Source 1 entities exactly.")

        for s1_id, cands in zip(cand_df["source1_entity_id"], cand_df["candidate_entity_ids"]):
            c_list = [c.strip() for c in cands.split(",") if c.strip()] if cands else []
            cand_dict[s1_id] = set(c_list)
        print(f"  [PASS] candidate_pairs.tsv structure and {len(cand_s1_ids):,} S1 rows verified.")
    except Exception as e:
        errors.append(f"Error parsing candidate_pairs.tsv: {e}")

    # Check 4: Validate matching_results.tsv
    print("\n3. Validating matching_results.tsv...")
    try:
        with open(matching_results_path, "r", encoding="utf-8") as f:
            header = f.readline().strip()
            if header != "source1_entity_id\tmatched_entity_ids":
                errors.append(
                    f"matching_results.tsv header mismatch. Expected: 'source1_entity_id\\tmatched_entity_ids', Found: '{header}'"
                )
    except Exception as e:
        errors.append(f"Could not read matching_results.tsv: {e}")

    total_matches = 0
    zero_matches_count = 0
    try:
        match_df = pd.read_csv(matching_results_path, sep="\t", dtype=str, keep_default_na=False)
        match_s1_ids = list(match_df["source1_entity_id"])
        if len(match_s1_ids) != len(expected_s1_ids):
            errors.append(
                f"matching_results.tsv row count ({len(match_s1_ids):,}) != expected ({len(expected_s1_ids):,})"
            )
        if set(match_s1_ids) != expected_s1_set:
            errors.append("matching_results.tsv S1 entity IDs do not match Test Source 1 entities exactly.")

        # Entity ID and duplicate checks
        duplicate_id_violations = 0
        s1_as_match_violations = 0
        candidate_containment_violations = 0

        for s1_id, matches in zip(match_df["source1_entity_id"], match_df["matched_entity_ids"]):
            m_list = [m.strip() for m in matches.split(",") if m.strip()] if matches else []
            total_matches += len(m_list)
            if not m_list:
                zero_matches_count += 1

            # Check duplicate IDs in row
            if len(m_list) != len(set(m_list)):
                duplicate_id_violations += 1

            # Check matching ID format (must be S2 or S3)
            for mid in m_list:
                if mid.startswith("S1-"):
                    s1_as_match_violations += 1
                if not (mid.startswith("S2-") or mid.startswith("S3-")):
                    errors.append(f"Invalid matched entity ID prefix '{mid}' for S1: {s1_id}")

            # Check candidate containment
            if s1_id in cand_dict:
                valid_cands = cand_dict[s1_id]
                for mid in m_list:
                    if mid not in valid_cands:
                        candidate_containment_violations += 1

        if duplicate_id_violations > 0:
            errors.append(f"Found {duplicate_id_violations} rows with duplicate matched IDs.")
        if s1_as_match_violations > 0:
            errors.append(f"Found {s1_as_match_violations} rows predicting S1 as matched ID (must be S2/S3).")
        if candidate_containment_violations > 0:
            errors.append(
                f"Found {candidate_containment_violations} predicted matches that are not in candidate_pairs.tsv."
            )

        print(f"  [PASS] matching_results.tsv structure and {len(match_s1_ids):,} S1 rows verified.")
        print(f"  Total predicted matches:     {total_matches:,}")
        print(f"  Zero-match S1 entities:       {zero_matches_count:,} ({zero_matches_count / len(match_s1_ids):.2%})")

    except Exception as e:
        errors.append(f"Error parsing matching_results.tsv: {e}")

    # Summary
    print("\n" + "=" * 60)
    if errors:
        print("VALIDATION RESULT: FAILED")
        print("=" * 60)
        for err in errors:
            print(f"  [X] {err}")
        return False
    else:
        print("VALIDATION RESULT: PASS — no blocking issues found. Safe to submit.")
        print("=" * 60)
        return True


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    matching_path = project_root / "output" / "matching_results.tsv"
    cand_path = project_root / "output" / "candidate_pairs.tsv"

    test_s1_clean = project_root / "processed" / "test_source1_clean.csv"
    if not test_s1_clean.exists():
        test_s1_clean = project_root / "test" / "test_source1.tsv"

    valid = validate_submission(matching_path, cand_path, test_s1_clean)
    sys.exit(0 if valid else 1)
