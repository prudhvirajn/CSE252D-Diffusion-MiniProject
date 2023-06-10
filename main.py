import argparse
import json

from model import ImageGen
from utils import image_grid, save_gif, downsample_img

def main(args):
    dataset_path = args.dataset_path 
    output_filepath = args.output_filepath
    output_gif_filepath = args.output_gif_filepath
    script_num = args.script_num
    gif_duration = args.gif_duration
    interp_steps = args.interp_steps
    prefix_id = args.prefix_num
    style_id = args.style_id
    
    with open(dataset_path, "r") as file:
        dataset = json.load(file)


    if script_num < 0 or script_num > len(dataset["scripts"]):
        print(f"Invalid script number. (Choose between [0,{len(dataset['scripts'])}]. Exiting..)")
        exit()
    elif prefix_id < 0 or prefix_id > len(dataset["prefixes"]):
        print(f"Invalid prefix number. (Choose between [0,{len(dataset['prefixes'])}]. Exiting..)")
        exit()
    elif style_id < 0 or style_id > len(dataset["suffixes"]):
        print(f"Invalid suffix number. (Choose between [0,{len(dataset['suffixes'])}]. Exiting..)")
        exit()
    elif interp_steps < 0:
        print("Invalid interpolation steps ([0,inf]). Exiting..")
        exit()
        
    negative_prompt = dataset["negative_prompt"]
    
    prefix = dataset["prefixes"][prefix_id]
    suffix = dataset["suffixes"][style_id]
    
    script = dataset["scripts"][script_num]
    
    prompts = [prefix + scene + suffix for scene in script]
    
    print("Intializing Image Generation models")
    pipeline = ImageGen()
    
    print("Generating key frames")
    key_frames = pipeline.generate_sequence(prompts, negative_prompt)

    for i in range(len(key_frames)):
        key_frames[i] = downsample_img(key_frames[i], (256,256))
    
    print("Interpolating story between key frames")
    story_board = pipeline.interpolate_story(key_frames)
    
    final_image = image_grid(story_board, 1, len(story_board))
    
    print("Saving")
    final_image.save(output_filepath)
    save_gif(story_board, output_gif_filepath, gif_duration)
    
if __name__ == '__main__':
    MAX_SCRIPT = 4
    MAX_PREFIX = 2
    MAX_SUFFIX = 2

    parser = argparse.ArgumentParser(description="Application description")
    parser.add_argument("--script_num", "-s", dest="script_num", type=int, action="store", default=0, required=False, help=f"Choose a script number [0,{MAX_SCRIPT}]")
    parser.add_argument("--duration", "-d", dest="gif_duration", type=int, action="store", default=300, required=False, help="Duration (ms) for each frame in gif")
    parser.add_argument("--interpolation_steps", "-i", dest="interp_steps", type=int, action="store", default=0, required=False, help="Number of interpolation steps between story keypoints")
    parser.add_argument("--prefix_num", "-p", dest="prefix_num", type=int, action="store", default=0, required=False, help=f"Choose a prefix number for the text prompt [0,{MAX_PREFIX}]")
    parser.add_argument("--suffix_num", "-suf", dest="style_id", type=int, action="store", default=0, required=False, help="Choose a suffix number for the text prompt [0,{MAX_SUFFIX}]")
    parser.add_argument("--dataset_path", "-dp", dest="dataset_path", type=str, action="store", default="dataset.json", required=False, help="Path to dataset json file")
    parser.add_argument("--output_filepath", "-o", dest="output_filepath", type=str, action="store", default="output.jpg", required=False, help="Path to output image file")
    parser.add_argument("--output_gif_filepath", "-og", dest="output_gif_filepath", type=str, action="store", default="output.gif", required=False, help="Path to output gif file")

    args = parser.parse_args()

    if args.script_num < 0 or args.script_num > MAX_SCRIPT:
      print(f"Invalid script number. (Choose between [0,{MAX_SCRIPT}]. Exiting..)")
      exit()
    elif (args.gif_duration < 0):
      print("Invalid gif duration ([0,inf]). Exiting..")
      exit()

    main(args)
