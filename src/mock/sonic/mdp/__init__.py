"""SONIC tracking mdp — merges mjlab tracking mdp with shared + sonic terms."""

from mjlab.tasks.tracking.mdp import *  # noqa: F401, F403

from mock.mdp import (  # noqa: F401
    FutureMotionCommand,
    FutureMotionCommandCfg,
)
from mock.sonic.mdp.observations import sonic_g1_tokenizer  # noqa: F401
