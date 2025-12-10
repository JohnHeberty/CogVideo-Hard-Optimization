CogVideoX Optimizations Documentation
======================================

Welcome to the CogVideoX Optimizations documentation! This project provides
production-ready optimizations for running CogVideoX models on consumer GPUs
(e.g., RTX 3090 24GB) without OOM crashes.

Key Features
-----------

* **VRAM Optimization**: Reduce memory usage from 27GB to 0-18GB
* **Performance Boost**: 2-3x faster inference with intelligent CPU offload
* **Auto FPS Detection**: Correct video timing for CogVideoX (8fps) vs CogVideoX1.5 (16fps)
* **Motion Presets**: 5 quality presets for different motion types
* **Lazy Loading**: On-demand pipeline loading to minimize idle VRAM

Quick Start
----------

Install dependencies:

.. code-block:: bash

   pip install -r CogVideo/requirements.txt

Run CLI demo with optimizations:

.. code-block:: bash

   python3 CogVideo/inference/cli_demo.py \
     --prompt "A golden retriever sprinting" \
     --model_path THUDM/CogVideoX-5b \
     --motion_preset high_motion

Launch web interface:

.. code-block:: bash

   python3 CogVideo/inference/gradio_web_demo.py

User Guides
----------

.. toctree::
   :maxdepth: 2
   :caption: Guides

   guides/quickstart
   guides/motion_presets
   guides/vram_optimization
   guides/troubleshooting

API Reference
------------

.. toctree::
   :maxdepth: 2
   :caption: API

   api/fps_utils
   api/vram_utils
   api/motion_presets
   api/pipeline_utils

Advanced Topics
--------------

.. toctree::
   :maxdepth: 2
   :caption: Advanced

   advanced/performance_tuning
   advanced/custom_presets
   advanced/benchmarking

Migration & Changelog
-------------------

.. toctree::
   :maxdepth: 1
   :caption: Migration

   migration/changelog
   migration/from_original

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
