# Configuration for Faster RCNN training

# Dataset settings
DATA_ROOT = "./data/street_objects"
CLASSES = ["background", "Person", "Car", "Truck", "Bicycle", "Traffic light"]
NUM_CLASSES = 6

# Training settings
BATCH_SIZE = 2
NUM_EPOCHS = 24
LEARNING_RATE = 0.005
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0001
LR_STEPS = [16, 22]  # epochs to reduce lr
LR_GAMMA = 0.1

# Model settings
BACKBONE = "resnet50"
ANCHOR_SIZES = [32, 64, 128, 256, 512]
ANCHOR_RATIOS = [0.5, 1.0, 2.0]

# RPN settings
RPN_NMS_THRESH = 0.7
RPN_POS_THRESH = 0.7
RPN_NEG_THRESH = 0.3

# Detection settings
SCORE_THRESH = 0.05
NMS_THRESH = 0.5
MAX_DETECTIONS = 100

# Other
NUM_WORKERS = 4
DEVICE = "cuda"
OUTPUT_DIR = "./output"
SEED = 42
