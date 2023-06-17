from enum import Enum


class Constants(Enum):
    FRAME_SKIP = 8
    GAMMA = 0.999
    AGENTS_PER_MATCH = 2
    NUM_INSTANCES = 10
    TARGET_STEPS = 750_000
    BATCH_SIZE = 50_000
    TRAINING_INTERVAL = 20_000_000_000
    CONFIG_NAME = "RLBOT_config"
