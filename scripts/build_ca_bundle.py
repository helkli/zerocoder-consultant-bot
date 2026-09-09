"""Собирает итоговый CA bundle для бота: стандартные корни (certifi) +
локальный сертификат антивируса/прокси из certs/ca_store.pem.

Результат сохраняется в certs/ca.pem. Его и нужно указывать в SSL_CA_BUNDLE.
"""

import sys
from pathlib import Path

import certifi

BASE_DIR = Path(__file__).resolve().parent.parent
STORE = BASE_DIR / "certs" / "ca_store.pem"
OUT = BASE_DIR / "certs" / "ca.pem"

if not STORE.is_file():
    print(f"Ошибка: не найден {STORE}. Сначала выполните scripts/get_ca.ps1")
    sys.exit(1)

part1 = Path(certifi.where()).read_text(encoding="utf-8")
part2 = STORE.read_text(encoding="utf-8")

OUT.write_text(part1 + "\n" + part2, encoding="utf-8")
print(f"Сохранено: {OUT}")
print(f"  стандартные корни (certifi):   {len(part1)} символов")
print(f"  локальный CA (ca_store.pem):   {len(part2)} символов")