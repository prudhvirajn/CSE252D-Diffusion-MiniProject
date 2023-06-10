import argparse
import json

from model import ImageGen
from utils import image_grid

def main():
    dataset_path = "dataset.json"
    output_filepath = "output.jpg"
    
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
        
    negative_prompt = dataset["negative_prompt"]
    
    prefix = dataset["prefixes"][0]
    suffix = dataset["suffixes"][0]
    
    script = dataset["scripts"][0]
    
    prompts = [prefix + scene + suffix for scene in script]
    
    print("Intializing Image Generation models")
    pipeline = ImageGen()
    
    print("Generating key frames")
    key_frames = pipeline.generate_sequence(prompts, negative_prompt)
    
    print("Interpolating story between key frames")
    story_board = pipeline.interpolate_story(key_frames)
    
    final_image = image_grid(story_board, 1, len(story_board))
    
    final_image.save(output_filepath)
    
if __name__ == '__main__':
    main()
