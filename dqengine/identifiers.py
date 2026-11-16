"""Identifier validation helpers built from public standards.

- LEI: ISO 17442 (20 chars, last 2 are check digits, ISO 7064 MOD 97-10)
- ISIN: ISO 6166 (12 chars, last char is a Luhn check digit)
- CONCAT: MiFIR RTS 22 national client identifier fallback
"""
from __future__ import annotations

import re
from datetime import datetime

LEI_PATTERN = re.compile(r"^[A-Z0-9]{18}[0-9]{2}$")
ISIN_PATTERN = re.compile(r"^[A-Z]{2}[A-Z0-9]{9}[0-9]$")
CONCAT_PATTERN = re.compile(r"^[A-Z]{2}\d{8}[A-Z#]{5}[A-Z#]{5}$")
UTC_TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d{1,6})?Z$")


def _alnum_to_digits(value: str) -> str:
    """Convert letters to numbers (A=10 ... Z=35), leave digits as they are."""
    return "".join(str(int(ch, 36)) for ch in value)


# ---------- LEI ----------
def lei_check_digits(base18: str) -> str:
    """Compute the two LEI check digits for an 18-character base."""
    num = int(_alnum_to_digits(base18.upper() + "00"))
    return f"{98 - (num % 97):02d}"


def is_valid_lei(value) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip().upper()
    if not LEI_PATTERN.match(value):
        return False
    return int(_alnum_to_digits(value)) % 97 == 1


# ---------- ISIN ----------
def isin_check_digit(base11: str) -> str:
    """Compute the ISIN check digit (Luhn over the letter-expanded string)."""
    digits = _alnum_to_digits(base11.upper())
    total = 0
    # Double every second digit starting from the rightmost.
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return str((10 - total % 10) % 10)


def is_valid_isin(value) -> bool:
    if not isinstance(value, str):
        return False
    value = value.strip().upper()
    if not ISIN_PATTERN.match(value):
        return False
    return isin_check_digit(value[:11]) == value[11]


# ---------- CONCAT / timestamps ----------
def is_valid_concat(value) -> bool:
    if not isinstance(value, str) or not CONCAT_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value[2:10], "%Y%m%d")
    except ValueError:
        return False
    return True


def is_utc_timestamp(value) -> bool:
    if not isinstance(value, str) or not UTC_TS_PATTERN.match(value):
        return False
    try:
        datetime.strptime(value[:19], "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return False
    return True
