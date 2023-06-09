from PIL import Image
import numpy as np
import torch

def slerp(val, low, high):
    """
    Find the interpolation point between the 'low' and 'high' values for the given 'val'. See https://en.wikipedia.org/wiki/Slerp for more details on the topic.
    """
    low_norm = low / torch.norm(low)
    high_norm = high / torch.norm(high)
    omega = torch.acos((low_norm * high_norm))
    so = torch.sin(omega)
    res = (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high
    return res

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[1].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        resized_img = downsample_img(img, imgs[1].size)
        grid.paste(resized_img, box=(i%cols*w, i//cols*h))
    return grid

def save_gif(frames, output_path, duration=300):
    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    
def downsample_img(story_keypoint, size):
    return story_keypoint.resize(size, Image.LANCZOS)