"""Checkpoint helpers used by read-only Cosmos evaluation jobs."""

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2.checkpointer.dcp import DistributedCheckpointer


class LoadOnlyDistributedCheckpointer(DistributedCheckpointer):
    """Load a DCP checkpoint normally, but never write a probe checkpoint."""

    def save(self, *args, **kwargs) -> None:
        iteration = kwargs.get("iteration", "unknown")
        log.info(f"Probe mode: skipping checkpoint save at iteration {iteration}.")
