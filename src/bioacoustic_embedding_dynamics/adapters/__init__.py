"""Convert bacpipe or BMZ BirdNET outputs to JSONL detection manifests."""

from .bacpipe_adapter import bacpipe_loader_to_manifest
from .bmz_adapter import bmz_birdnet_to_manifest

__all__ = ["bacpipe_loader_to_manifest", "bmz_birdnet_to_manifest"]
