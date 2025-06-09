import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(PROJECT_ROOT, 'data', 'train')
test_dir = os.path.join(PROJECT_ROOT, 'data', 'test')
image_classes = os.listdir(train_dir)


