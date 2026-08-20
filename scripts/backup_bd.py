#!/usr/bin/env python3
"""scripts/backup_bd.py — Backup y restauración de la BD de perfiles (SEC-001).

Uso:
    python scripts/backup_bd.py backup [--destino RUTA]
    python scripts/backup_bd.py restore RUTA_BACKUP

El backup usa la API de sqlite3 (segura con WAL) y queda con permisos 600,
igual que la BD. La restauración sobrescribe data/climasafe.db con el
contenido del backup; hazla con el bot y la web detenidos.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

from climasafeai.db.manager import DBManager


def _gestor() -> DBManager:
    return DBManager()  # data/climasafe.db por defecto


def cmd_backup(args: argparse.Namespace) -> int:
    db = _gestor()
    if not db.db_path.exists():
        print(f"La BD {db.db_path} no existe; nada que respaldar.")
        return 1
    destino = Path(args.destino) if args.destino else (
        Path("data/backups") / f"climasafe_{datetime.now():%Y%m%d_%H%M%S}.db"
    )
    info = db.backup(destino)
    print(f"Backup creado: {info['backup']}")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    db = _gestor()
    info = db.restaurar(args.origen)
    print(f"BD restaurada desde: {info['restaurado']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_backup = sub.add_parser("backup", help="crea una copia de seguridad")
    p_backup.add_argument(
        "--destino",
        help="ruta del fichero de backup (por defecto data/backups/climasafe_AAAAMMDD_HHMMSS.db)",
    )
    p_backup.set_defaults(func=cmd_backup)
    p_restore = sub.add_parser("restore", help="restaura la BD desde una copia")
    p_restore.add_argument("origen", help="ruta del fichero de backup a restaurar")
    p_restore.set_defaults(func=cmd_restore)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())