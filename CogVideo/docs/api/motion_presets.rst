Motion Presets
==============

The ``motion_presets`` module provides quality presets for different motion types.

.. automodule:: motion_presets
   :members:
   :undoc-members:
   :show-inheritance:

Preset Functions
---------------

get_preset
~~~~~~~~~

.. autofunction:: get_preset

Example:

.. code-block:: python

   from motion_presets import get_preset

   preset = get_preset("high_motion")
   print(preset.guidance_scale)  # 6.5

list_presets
~~~~~~~~~~~

.. autofunction:: list_presets

apply_preset_to_pipeline_args
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: apply_preset_to_pipeline_args

Example:

.. code-block:: python

   from motion_presets import apply_preset_to_pipeline_args

   args = apply_preset_to_pipeline_args("high_motion", {
       "prompt": "A golden retriever sprinting",
       "num_frames": 49
   })

Available Presets
----------------

balanced
~~~~~~~

Default balanced preset for general use.

* guidance_scale: 6.0
* num_inference_steps: 50

fast
~~~~

Fast preview mode (40% faster).

* guidance_scale: 5.0
* num_inference_steps: 30

quality
~~~~~~

High quality for final renders.

* guidance_scale: 7.0
* num_inference_steps: 75

high_motion
~~~~~~~~~~

Optimized for fast action and sports (fixes "golden retriever" issue).

* guidance_scale: 6.5
* num_inference_steps: 60

subtle
~~~~~

Gentle, smooth motion.

* guidance_scale: 5.5
* num_inference_steps: 55
