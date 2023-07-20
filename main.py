import os

from PIL import Image
from wmdetection.models import get_watermarks_detection_model
from wmdetection.pipelines.predictor import WatermarksPredictor
import time
from pillow_heif import register_heif_opener

register_heif_opener()

# checkpoint is automatically downloaded
model, transforms = get_watermarks_detection_model(
    'convnext-tiny',
    fp16=False,
    cache_dir='./model-weights/'
)
predictor = WatermarksPredictor(model, transforms, 'cuda:0')

img = Image.open('images/watermark/8.jpg')
s = time.time()
result = predictor.predict_image(img)
print(f"time: {time.time() - s}")
print('watermarked' if result else 'clean') # prints "watermarked"

# Multi
s = time.time()
with_files = os.listdir("./images/laion/with/")
with_files_d = ["images/laion/with/" + f for f in with_files]

results = predictor.run(with_files_d[:64], num_workers=1, bs=64)
end = time.time() - s
print(f"time: {end} results: {results}")
os.mkdir("./images/wrong5")
import shutil
for i, result in enumerate(results):
    if result == 1:
        shutil.copy2("./images/laion/with/" + with_files[i], "./images/wrong5/" + with_files[i])
        # print(with_files[i])
    # print('watermarked' if result else 'clean')


