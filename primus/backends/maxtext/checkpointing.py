from typing import Any
from etils import epath
import orbax.checkpoint as ocp

from MaxText import max_logging


def create_orbax_checkpoint_manager(
    checkpoint_dir: str,
    enable_checkpointing: bool,
    use_async: bool,
    save_interval_steps: int,
    dataset_type: None | str = "tfds",
    orbax_logger: Any = None,  # pytype: disable=attribute-error
    use_ocdbt: bool = True,
    use_zarr3: bool = True,
    max_to_keep: int = 5,
):
  """Returns specified Orbax (async or not) CheckpointManager or None if checkpointing is disabled."""
  if not enable_checkpointing:
    max_logging.log("Checkpointing disabled, not creating checkpoint manager.")
    return None

  max_logging.log(f"Creating checkpoint manager with ocdbt={use_ocdbt} and zarr3={use_zarr3}")

  if dataset_type == "grain":
    item_names = ("items", "iter")
  else:
    item_names = ("items",)

  # local storage checkpoint needs parent directory created
  p = epath.Path(checkpoint_dir)
  p.mkdir(exist_ok=True, parents=True)
  # we need to use ocdbt and zarr3 to control max file size in the checkpoint
  # omitting `iter` uses default handler for `iter`
  item_handlers = {"items": ocp.PyTreeCheckpointHandler(use_ocdbt=use_ocdbt, use_zarr3=use_zarr3)}
  manager = ocp.CheckpointManager(
      p,
      item_names=item_names,
      item_handlers=item_handlers,
      options=ocp.CheckpointManagerOptions(
          create=True,
          save_interval_steps=save_interval_steps,
          enable_async_checkpointing=use_async,
          max_to_keep = max_to_keep,
      ),
      logger=orbax_logger,
  )

  max_logging.log("Checkpoint manager created!")
  return manager
