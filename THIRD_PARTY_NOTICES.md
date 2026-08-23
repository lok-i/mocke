# Third-party notices

The BSD-3-Clause license at the repository root applies only to original Mocke
material. The following components retain their upstream terms.

## GEAR-SONIC

The SONIC-compatible observation, action, environment, export, and checkpoint
porting code under `src/mocke/sonic/`, the relevant shared contracts under
`src/mocke/mdp/`, and `scripts/port_sonic_checkpoint.py` implement or adapt
GEAR-SONIC interfaces. They are subject to Apache-2.0 in addition to Mocke's
license. See `LICENSES/Apache-2.0.txt`.

Files under `pretrained/sonic/` are GEAR-SONIC model weights or derivative
models. They are subject to the NVIDIA Open Model License. Redistribution must
include that agreement and this attribution:

> Licensed by NVIDIA Corporation under the NVIDIA Open Model License.

See `LICENSES/GEAR-SONIC-LICENSE.txt` and the accompanying GEAR-SONIC notices
in `LICENSES/`. Source: [NVIDIA GR00T Whole-Body
Control](https://github.com/NVlabs/GR00T-WholeBodyControl).

## TextOp

The TextOp-compatible implementation under `src/mocke/textop/` and the model
artifacts under `pretrained/textop/` originate from or adapt the TextOp project.
They retain the TextOp Team's MIT terms in `LICENSES/TextOp-MIT.txt`.

Source: [TeleHuman/TextOp](https://github.com/TeleHuman/TextOp).

## Runtime dependencies

Mocke imports but does not vendor `mjlab`, RSL-RL, MuJoCo, PyTorch, NumPy, and
other installed dependencies. Their own distributions and licenses govern
their use.

Acknowledgement and citation do not replace compliance with any license above.
