from __future__ import annotations

from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm.auto import tqdm


class DownloadError(RuntimeError):
    pass


def filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    if not name:
        raise ValueError(f"Could not infer filename from URL: {url}")
    return name


def download_file(
    url: str,
    destination: str | Path,
    *,
    show_progress: bool = True,
    overwrite: bool = False,
    timeout: int = 60,
) -> Path:
    """Download one file with streaming progress.

    Existing files are reused by default.
    """
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 0 and not overwrite:
        return destination

    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DownloadError(f"Failed to download {url}: {exc}") from exc

    total = int(response.headers.get("content-length", 0))
    tmp_path = destination.with_suffix(destination.suffix + ".tmp")

    with tmp_path.open("wb") as f, tqdm(
        total=total if total > 0 else None,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {destination.name}",
        disable=not show_progress,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 256):
            if not chunk:
                continue
            f.write(chunk)
            bar.update(len(chunk))

    tmp_path.replace(destination)
    return destination


def ensure_files(
    files: dict[str, str],
    root: str | Path,
    *,
    show_progress: bool = True,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Ensure a mapping of logical names -> URLs exists under root."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for logical_name, url in files.items():
        paths[logical_name] = download_file(
            url,
            root / filename_from_url(url),
            show_progress=show_progress,
            overwrite=overwrite,
        )
    return paths
