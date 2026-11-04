"""Exercise RoboTwin's SAPIEN raster camera and verify the selected Vulkan binaries."""
import os
from pathlib import Path
import numpy as np
import sapien.core as sapien

engine = sapien.Engine()
renderer = sapien.SapienRenderer()
engine.set_renderer(renderer)
sapien.render.set_camera_shader_dir("default")
scene = engine.create_scene(sapien.SceneConfig())
scene.set_ambient_light([0.7, 0.7, 0.7])
builder = scene.create_actor_builder()
builder.add_box_visual(half_size=[0.1, 0.1, 0.1], material=[0.8, 0.1, 0.1])
builder.build_static(name="cube")
camera = scene.add_camera("preflight", 64, 48, 1.0, 0.01, 10)
camera.set_pose(sapien.Pose([-1, 0, 0]))
scene.update_render()
camera.take_picture()
rgb = camera.get_picture("Color")[..., :3]
if rgb.shape != (48, 64, 3) or not np.isfinite(rgb).all() or rgb[..., 0].std() < 0.01:
    raise RuntimeError("SAPIEN returned a blank or invalid RGB frame")
root = Path(os.environ['TOKENWORLD_RENDER_LIB']).resolve()
loaded = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()}
for library in ('libvulkan.so.1', 'libvulkan_lvp.so'):
    if str(root / library) not in loaded:
        raise RuntimeError(f"Expected renderer library was not loaded: {root / library}")
print("Renderer preflight passed: SAPIEN captured a 64x48 RGB frame using the bundled software Vulkan driver.", flush=True)
