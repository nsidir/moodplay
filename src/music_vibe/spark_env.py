"""Create a local SparkSession that works on Windows and Linux without manual setup."""

from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path

from .config import ARTIFACTS

_JAVA = "java.exe" if os.name == "nt" else "java"
# Spark 4 needs Java 17 or 21. Searched when JAVA_HOME is unset or points at a non-JDK.
_JDK_DIRS = (
    [Path(r"C:\Program Files\Java"), Path(r"C:\Program Files\Eclipse Adoptium"),
     Path(r"C:\Program Files\Microsoft")]
    if os.name == "nt"
    else [Path("/usr/lib/jvm"), Path("/usr/lib64/jvm"), Path("/opt/java"),
          Path.home() / ".sdkman" / "candidates" / "java"]
)


def _is_jdk(d: Path) -> bool:
    return (d / "bin" / _JAVA).exists()


def _find_java_home() -> str | None:
    cur = os.environ.get("JAVA_HOME")
    if cur and _is_jdk(Path(cur)):
        return cur
    # Oracle's "javapath" shim on Windows and /usr/bin/java on Linux are not JDK homes.
    for base in _JDK_DIRS:
        if base.is_dir():
            for pref in ("21", "17"):
                # jdk-21.0.2, java-21-openjdk-amd64, temurin-21-jdk, 21.0.2-tem (sdkman); not jdk-11.0.21
                pat = re.compile(rf"(?:^|jdk-?|java-|temurin-|openjdk-?){pref}(?!\d)", re.I)
                cands = [d for d in base.iterdir() if d.is_dir() and pat.search(d.name) and _is_jdk(d)]
                if cands:
                    return str(sorted(cands, reverse=True)[0])
    # Fall back to whatever `java` is on PATH (resolve symlinks like /etc/alternatives).
    exe = shutil.which("java")
    if exe:
        home = Path(exe).resolve().parent.parent
        if _is_jdk(home):
            return str(home)
    return cur


def get_spark(app: str = "music-vibe", driver_memory: str = "3g"):
    java_home = _find_java_home()
    if java_home:
        os.environ["JAVA_HOME"] = java_home
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    from pyspark.sql import SparkSession

    tmp = ARTIFACTS / "spark-tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    spark = (
        SparkSession.builder.master("local[*]")
        .appName(app)
        .config("spark.driver.memory", driver_memory)
        # avoid hostname/loopback resolution problems (common on Linux and on VPNs)
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.local.dir", str(tmp))
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    return spark
