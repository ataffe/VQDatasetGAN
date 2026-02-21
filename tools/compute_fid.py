import torch
from ignite.engine import Engine
from ignite.metrics.gan import FID

# 1. Define your FID metric
# By default, it uses a pretrained InceptionV3 from torchvision
fid_metric = FID(device="cuda" if torch.cuda.is_available() else "cpu")

# 2. Define an evaluation step
def evaluation_step(engine, batch):
    # 'batch' should contain (generated_images, real_images)
    # Ensure they are in the range [0, 1] and 3-channel
    gen_imgs, real_imgs = batch
    return gen_imgs, real_imgs

# 3. Create the evaluator and attach the metric
evaluator = Engine(evaluation_step)
fid_metric.attach(evaluator, "fid")

# 4. Run evaluation
# Your data_loader should yield pairs of (fake_images, real_images)
state = evaluator.run(validation_loader)
print(f"FID Score: {state.metrics['fid']}")