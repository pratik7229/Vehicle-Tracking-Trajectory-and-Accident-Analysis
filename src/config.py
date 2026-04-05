# Defination of all global variables goes here 
import torch

# Device Settings
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


# YOLO Settings
MODEL_PATH = "yolo26l.pt" #"yolo11m.pt" 
CONFIDENCE_THRESHOLD = 0.1
VEHICLE_CLASSES = [2, 3, 5, 7] # Car, Motorbike, Bus, Truck
DATASET_PATH = "/Users/pratik/Documents/Projects/kaggleCompetation_data"


# Deep Sort Settings
MAX_AGE = 30
N_INT = 4
MAX_COSINE_DISTANCE = 0.3

# Analytics Settings
IOU_THRESHOLD = 0.01
ACCEL_THRESHOLD = 50.0  # Pixels/frame^2 - adjust based on resolution
HISTORY_LIMIT = 20      # Frames to keep in memory per ID
