"""Test architetturale: fa rispettare le dipendenze tra i sottopacchetti.

Il codice di ogni file viene letto con il modulo ast (senza eseguirlo) per
ricavare gli import, che vengono confrontati con le regole dell'architettura.
Per semplicità tutti gli import interni devono essere assoluti
(from cubesat_sim.core import ...): gli import relativi sono segnalati.
"""

import ast
from pathlib import Path

PACKAGE = "cubesat_sim"
PACKAGE_DIR = Path(__file__).resolve().parents[1] / "src" / PACKAGE

# Sottopacchetti di dominio: ciascuno dipende solo da core.
DOMAIN = frozenset({"environment", "attitude", "resources", "modes"})

# Per ogni sottopacchetto, gli altri sottopacchetti che può importare.
# Importare moduli del proprio sottopacchetto è sempre ammesso.
ALLOWED_INTERNAL: dict[str, frozenset[str]] = {
    "core": frozenset(),
    "environment": frozenset({"core"}),
    "attitude": frozenset({"core"}),
    "resources": frozenset({"core"}),
    "modes": frozenset({"core"}),
    "simulation": DOMAIN | {"core"},
    "analysis": DOMAIN | {"core", "simulation"},
    "dashboard": DOMAIN | {"core", "simulation", "analysis"},
}

# Librerie esterne che solo la dashboard può importare.
DASHBOARD_ONLY = frozenset({"streamlit", "plotly"})


def imported_modules(source: str) -> list[str]:
    """Restituisce i nomi dei moduli importati nel codice sorgente."""
    modules: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                # Import relativo: "from ..core import x" diventa "..core".
                modules.append("." * node.level + (node.module or ""))
            elif node.module == PACKAGE:
                # "from cubesat_sim import core" importa il sottopacchetto core.
                modules.extend(f"{PACKAGE}.{alias.name}" for alias in node.names)
            elif node.module is not None:
                modules.append(node.module)
    return modules


def check_import(subpackage: str, module: str) -> str | None:
    """Restituisce il motivo della violazione, oppure None se l'import è ammesso."""
    if module.startswith("."):
        return f"import relativo non ammesso ({module})"
    parts = module.split(".")
    if parts[0] == PACKAGE:
        target = parts[1] if len(parts) > 1 else ""
        if target != subpackage and target not in ALLOWED_INTERNAL[subpackage]:
            return f"{subpackage} non può importare {module}"
    elif parts[0] in DASHBOARD_ONLY and subpackage != "dashboard":
        return f"{subpackage} non può importare {parts[0]}"
    return None


def find_violations(package_dir: Path) -> list[str]:
    """Controlla gli import di tutti i file dei sottopacchetti.

    Restituisce una riga per ogni violazione: lista vuota se è tutto in regola.
    """
    found: list[str] = []
    for path in sorted(package_dir.rglob("*.py")):
        relative = path.relative_to(package_dir)
        if len(relative.parts) == 1:
            # File del pacchetto principale: non appartiene a un sottopacchetto.
            continue
        name = relative.as_posix()
        subpackage = relative.parts[0]
        if subpackage not in ALLOWED_INTERNAL:
            found.append(f"{name}: sottopacchetto '{subpackage}' senza regole")
            continue
        for module in imported_modules(path.read_text(encoding="utf-8")):
            problem = check_import(subpackage, module)
            if problem is not None:
                found.append(f"{name}: {problem}")
    return found


def write_module(path: Path, source: str) -> None:
    """Scrive un file Python di prova, creando le cartelle necessarie."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def test_package_respects_architecture() -> None:
    assert PACKAGE_DIR.is_dir(), f"cartella non trovata: {PACKAGE_DIR}"
    assert find_violations(PACKAGE_DIR) == []


def test_checker_flags_violations(tmp_path: Path) -> None:
    # Controllo del controllore: un finto pacchetto con violazioni note.
    # Il codice di prova viene solo letto, mai eseguito.
    write_module(
        tmp_path / "environment" / "bad.py",
        "import streamlit\n"
        "from cubesat_sim.attitude import dynamics\n"
        "from ..core import config\n",
    )
    write_module(tmp_path / "telemetry" / "__init__.py", "")
    assert find_violations(tmp_path) == [
        "environment/bad.py: environment non può importare streamlit",
        "environment/bad.py: environment non può importare cubesat_sim.attitude",
        "environment/bad.py: import relativo non ammesso (..core)",
        "telemetry/__init__.py: sottopacchetto 'telemetry' senza regole",
    ]


def test_checker_accepts_allowed_imports(tmp_path: Path) -> None:
    write_module(
        tmp_path / "simulation" / "loop.py",
        "import numpy\n"
        "from cubesat_sim import attitude\n"
        "from cubesat_sim.core import config\n"
        "from cubesat_sim.simulation import state\n",
    )
    write_module(tmp_path / "dashboard" / "app.py", "import streamlit\n")
    assert find_violations(tmp_path) == []
