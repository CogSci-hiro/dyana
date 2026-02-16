from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional
import os
import uuid


class ConfigError(ValueError):
    """Raised when required runtime configuration is missing or invalid."""


def _parse_minimal_yaml(text: str) -> dict[str, Any]:
    """Parse a tiny subset of YAML used by DyANA config files."""

    data: dict[str, Any] = {}
    current_section: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue

        if not line.startswith(" "):
            if ":" not in line:
                continue
            key, value = [part.strip() for part in line.split(":", 1)]
            if value == "":
                data[key] = {}
                current_section = key
            else:
                data[key] = value.strip("\"'")
                current_section = None
            continue

        if current_section is None:
            continue
        child = line.strip()
        if ":" not in child:
            continue
        key, value = [part.strip() for part in child.split(":", 1)]
        section = data.setdefault(current_section, {})
        if isinstance(section, dict):
            section[key] = value.strip("\"'")
    return data


def load_config(root: Path) -> dict[str, Any]:
    """
    Load DyANA config from repository root if present.

    Search order:
    1) ``config.yaml``
    2) ``dyana.yaml``
    """

    for filename in ("config.yaml", "dyana.yaml"):
        config_path = root / filename
        if config_path.exists():
            return _parse_minimal_yaml(config_path.read_text())
    return {}


def resolve_out_dir(config: dict[str, Any], cli_out_dir: Path | None) -> Path:
    """
    Resolve output directory from CLI arg or config.

    CLI value has highest priority. Fallbacks:
    - ``paths.out_dir``
    - ``io.out_dir``
    """

    if cli_out_dir is not None:
        return cli_out_dir

    for section_name in ("paths", "io"):
        section = config.get(section_name)
        if isinstance(section, dict):
            out_dir = section.get("out_dir")
            if isinstance(out_dir, str) and out_dir.strip():
                return Path(out_dir.strip())

    raise ConfigError(
        "No out_dir provided. Pass --out-dir or create config.yaml with paths.out_dir."
    )


@dataclass(frozen=True)
class ErrorHandlingConfig:
    """
    Configuration for error-handling + logging behavior.

    Parameters
    ----------
    mode
        "debug" re-raises immediately on failure; "run" records failures and continues
        when it still makes sense (skipping dependent steps).
    log_dir
        Directory where log files and JSONL event logs are written.
    run_id
        Unique identifier for the run. If "auto", a UUID4 is generated.
    console_level
        Logging level for console output.
    file_level
        Logging level for file output.
    write_jsonl
        If True, writes structured JSONL events to <log_dir>/events_<run_id>.jsonl.
    max_failures
        If set, stops scheduling new work once the number of *failed* steps reaches this value.
        In "debug" mode this is ignored (you stop at first failure anyway).
    treat_warnings_as_errors
        Optional knob if you later want to elevate warnings to failures in a single place.
    env_prefix
        If you want environment-variable overrides, set a prefix like "PALIMPS_" or "DYANA_".

    Usage example
    -------------
        cfg = ErrorHandlingConfig(mode="run", log_dir=Path("logs"))
    """

    mode: Literal["debug", "run"] = "run"
    log_dir: Path = Path("logs")
    run_id: str = "auto"

    console_level: int = 20  # logging.INFO
    file_level: int = 10  # logging.DEBUG

    write_jsonl: bool = True
    max_failures: Optional[int] = None
    treat_warnings_as_errors: bool = False

    env_prefix: str = field(default="", repr=False)

    def resolved_run_id(self) -> str:
        """Return a non-auto run id."""
        if self.run_id != "auto":
            return self.run_id
        return uuid.uuid4().hex[:10]

    @classmethod
    def from_env(cls, *, default: Optional["ErrorHandlingConfig"] = None) -> "ErrorHandlingConfig":
        """
        Create config from environment variables.

        Supported variables (prefix controlled by env_prefix on `default`):
        - <PFX>ERROR_MODE: "debug" | "run"
        - <PFX>LOG_DIR: path
        - <PFX>WRITE_JSONL: "1"/"0"
        - <PFX>MAX_FAILURES: integer

        Notes
        -----
        If `default` is None, uses cls() and prefix "".

        Usage example
        -------------
            cfg = ErrorHandlingConfig.from_env(default=ErrorHandlingConfig(env_prefix="DYANA_"))
        """
        base = default if default is not None else cls()
        pfx = base.env_prefix

        mode = os.getenv(f"{pfx}ERROR_MODE", base.mode).strip().lower()
        if mode not in ("debug", "run"):
            mode = base.mode

        log_dir = Path(os.getenv(f"{pfx}LOG_DIR", str(base.log_dir)))

        write_jsonl_raw = os.getenv(f"{pfx}WRITE_JSONL", "1" if base.write_jsonl else "0").strip()
        write_jsonl = write_jsonl_raw not in ("0", "false", "False", "")

        max_failures_raw = os.getenv(f"{pfx}MAX_FAILURES", "")
        max_failures = base.max_failures
        if max_failures_raw.strip():
            try:
                max_failures = int(max_failures_raw)
            except ValueError:
                max_failures = base.max_failures

        return cls(
            mode=mode,  # type: ignore[arg-type]
            log_dir=log_dir,
            run_id=base.run_id,
            console_level=base.console_level,
            file_level=base.file_level,
            write_jsonl=write_jsonl,
            max_failures=max_failures,
            treat_warnings_as_errors=base.treat_warnings_as_errors,
            env_prefix=pfx,
        )
