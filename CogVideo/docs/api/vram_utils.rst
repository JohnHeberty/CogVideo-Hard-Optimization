VRAM Utils
==========

The ``vram_utils`` module provides VRAM monitoring, offload strategy selection, and VAE configuration.

.. automodule:: vram_utils
   :members:
   :undoc-members:
   :show-inheritance:

Memory Monitoring
----------------

get_gpu_memory_info
~~~~~~~~~~~~~~~~~~

.. autofunction:: get_gpu_memory_info

Example:

.. code-block:: python

   from vram_utils import get_gpu_memory_info

   total_gb, used_gb, available_gb = get_gpu_memory_info()
   print(f"{available_gb:.1f}GB available")

log_vram_status
~~~~~~~~~~~~~~

.. autofunction:: log_vram_status

check_vram_availability
~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: check_vram_availability

Offload Strategies
-----------------

get_recommended_offload_strategy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: get_recommended_offload_strategy

Example:

.. code-block:: python

   from vram_utils import get_recommended_offload_strategy

   strategy = get_recommended_offload_strategy("THUDM/CogVideoX-5b")
   print(strategy)  # "model" on RTX 3090

apply_offload_strategy
~~~~~~~~~~~~~~~~~~~~

.. autofunction:: apply_offload_strategy

VAE Configuration
----------------

configure_vae_tiling
~~~~~~~~~~~~~~~~~~~

.. autofunction:: configure_vae_tiling

Example:

.. code-block:: python

   from vram_utils import configure_vae_tiling

   configure_vae_tiling(pipe, enable=True, tile_sample_min_height=256)

VRAM Requirements
----------------

.. autodata:: VRAM_REQUIREMENTS
   :annotation:

Estimated VRAM requirements per model:

* CogVideoX-2B: 8GB
* CogVideoX-5B: 17GB
* CogVideoX-5B-I2V: 18GB
* CogVideoX1.5-5B: 20GB
