Pipeline Utils
==============

The ``pipeline_utils`` module provides centralized pipeline loading with automatic optimizations.

.. automodule:: pipeline_utils
   :members:
   :undoc-members:
   :show-inheritance:

Pipeline Loading
---------------

load_pipeline
~~~~~~~~~~~~

.. autofunction:: load_pipeline

Example:

.. code-block:: python

   from pipeline_utils import load_pipeline

   pipe = load_pipeline("THUDM/CogVideoX-5b", "t2v", apply_optimizations=True)

load_shared_pipeline
~~~~~~~~~~~~~~~~~~~

.. autofunction:: load_shared_pipeline

Model Information
----------------

get_model_info
~~~~~~~~~~~~~

.. autofunction:: get_model_info

Example:

.. code-block:: python

   from pipeline_utils import get_model_info

   info = get_model_info("THUDM/CogVideoX-5b")
   print(info)
   # {
   #     "type": "t2v",
   #     "default_fps": 8,
   #     "default_frames": 49,
   #     "vram_gb": 17.0
   # }

get_pipeline_class
~~~~~~~~~~~~~~~~~

.. autofunction:: get_pipeline_class

validate_model_path
~~~~~~~~~~~~~~~~~~

.. autofunction:: validate_model_path
