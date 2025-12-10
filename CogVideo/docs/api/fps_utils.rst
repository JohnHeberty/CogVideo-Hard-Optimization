FPS Utils
=========

The ``fps_utils`` module provides automatic FPS detection and validation for CogVideoX models.

.. automodule:: fps_utils
   :members:
   :undoc-members:
   :show-inheritance:

Key Functions
------------

get_correct_fps
~~~~~~~~~~~~~~

.. autofunction:: get_correct_fps

Example:

.. code-block:: python

   from fps_utils import get_correct_fps

   fps = get_correct_fps("THUDM/CogVideoX-5b", num_frames=49)
   print(fps)  # 8

validate_fps_for_model
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: validate_fps_for_model

Example:

.. code-block:: python

   from fps_utils import validate_fps_for_model

   is_valid, message = validate_fps_for_model("THUDM/CogVideoX-5b", fps=8)
   print(message)  # "FPS 8 is correct for this model"

FPS Mapping
----------

.. autodata:: FPS_MAP
   :annotation:

The mapping between model families and their correct FPS values:

* CogVideoX (2B, 5B, 5B-I2V): 8 fps
* CogVideoX1.5 (5B, 5B-I2V): 16 fps
