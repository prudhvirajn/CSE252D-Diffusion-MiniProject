# CSE252D-Diffusion-MiniProject

This is the mini-project for CSE 252D Advanced Computer Vision (Spring 2023). We implement a pipeline to generate story-boards from a start prompt and end prompt.

One can run the entire pipeline on Colab here: https://colab.research.google.com/github/prudhvirajn/CSE252D-Diffusion-MiniProject/blob/main/CSE252D_Story_Diffusion.ipynb

If one does not prefer Colab, they can simply run the CSE252D_Story_Diffusion.ipynb notebook locally just comment out `!git clone https://github.com/prudhvirajn/CSE252D-Diffusion-MiniProject.git`. If they wish to customize individual parts, please refer to the documentation below. 

## Table of Contents
1. [Generating Intermediate Prompts using ChatGPT][1]
2. [Structure of dataset.json][2]
3. [Installing dependencies][3]
4. [Generating Story Board][4]

[1]: https://github.com/prudhvirajn/CSE252D-Diffusion-MiniProject#generating-intermediate-prompts-using-chatgpt
[2]: https://github.com/prudhvirajn/CSE252D-Diffusion-MiniProject/tree/main#dataset.json
[3]: https://github.com/prudhvirajn/CSE252D-Diffusion-MiniProject/tree/main#installing
[4]: https://github.com/prudhvirajn/CSE252D-Diffusion-MiniProject/tree/main#generating-story-board

## Generating Intermediate Prompts using ChatGPT

We provide a series of prompt sequences in the `dataset.json` file but if one wants to imrpovise they can do the following. 

Given a start and end text prompt, we use ChatGPT to generate intermediate prompts that form a possible explanation of how the start scene relates to the end scene. Essentially, we use ChatGPT to form a script that we will render using Diffusion pipelines. 

Unfortunately, ChatGPT API requires a paid account to generate keys. Regrettably we are unable to provide an automated script to generate the intermediate prompts. 

Pass in the Input:
```
Example: 
1. "Man walks towards car"
2. "Man opens the car door"
3. "Man sits in the car"
4. "Car drives away"

Write me another sequence of actions like above that starts with "{start_prompt}" and ends with "{end_prompt}"
```
<img width="343" alt="Screenshot 2023-06-11 at 12 05 31 PM" src="https://github.com/prudhvirajn/CSE252D-Diffusion-MiniProject/assets/7262241/f85ab8a7-caf6-4ac6-9d69-dc19e697c517">

We store the above in a json format noted below. 

## Structure of dataset.json
The file "dataset.json" stores information passed into the stable diffusion model to generate images. It contains following fields:

```
{
 "negative_prompt": a universal list of keywords telling the model what not to generate. This filters out most low quality and unwanted images.,
 "scripts": [a list of sequences of prompts telling the model what to generate for each key frame. Each sequence is generated by ChatGPT as mentioned above.],
 "prefixes": [a list of descriptive keywords used to control the texture],
 "suffixes": [a list of color styleS]
}
```

## Installing

To install dependencies, simply run

`python3 -m pip install -r requirements.txt`

## Generating Story Board

Default usage to generate storyboard from `dataset.json` file and output `output.jpg` and `output.gif` files:

`python3 main.py --script_num {Choose one of the scripts 0-4} -suf {Choose style id 0-1}`

Custom usage:

`python3 main.py --script_num {Choose one of the scripts} -suf {Choose style id} --dataset_path {Dataset json path} --output_filepath {Filepath to write out compiled storyboard png} --output_gif_filepath {Path to write out gif file}`


