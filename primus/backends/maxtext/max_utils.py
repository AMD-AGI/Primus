import os
import socket
import json

import jax

from MaxText import max_logging


def print_system_information():
  """Print system information of the current environment.
  Note that this will initialize the JAX backend."""
  max_logging.log(f"System Information: Jax Version: {jax.__version__}")
  max_logging.log(f"System Information: Jaxlib Version: {jax.lib.__version__}")
  max_logging.log(f"System Information: Jax Backend: {jax.extend.backend.get_backend().platform_version}")

  devices = jax.devices()
  max_logging.log(f"System Information: Number of devices: {len(devices)}, jax path {jax.__file__}")
  for i, device in enumerate(devices):
    if device.local_hardware_id is not None:
      max_logging.log(
          f"System Information: Device {i}: {device.id} "
          f"(Local id: {device.local_hardware_id}, Process index: {device.process_index})"
      )


def save_device_information(config):
  """Convert device information to JSON format."""
  devices = jax.devices()
  device_info = {'hostname': socket.gethostname(), 'devices': []}

  for device in devices:
    if device.local_hardware_id is not None:
      info = {
          "id": device.id,
          "local_hardware_id": device.local_hardware_id,
          "process_index": device.process_index,
          "device_kind": device.device_kind,
          "platform_version": jax.extend.backend.get_backend().platform_version,
      }
      device_info['devices'].append(info)
  # Save to JSON file
  device_info_path = os.path.join(config.base_output_directory, "device_info.json")
  with open(device_info_path, "w") as f:
    json.dump(device_info, f, indent=4)