# Automated Mask on Facial Images


[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

- Create a dataset of masked images from a dataset containing facial images.
- Can be used for further training models for face recogition with masks on.

Dataset can be accessed here:

#### Cloning the repository
``` 
git clone https://github.com/Codee0101/Automated-Mask-on-Facial-Images.git
cd Automated-Mask-on-Facial-Images
```
#### Making Virtual environment for the project

Enter the name for the virtual environment in place of 'myenv'
```
python3 -m venv <myvenv>
source <myvenv>/bin/activate
```
#### Installing the required packages to our virtual environment

```
pip install -r requirements.txt
```
#### Further instructions
In the wearmask.py folder:
- Add the path of the dataset containing facial images on line 221; dataset_path.
- Create an empty folder for storing the new masked images.
- Add the path of this folder in front of save_dataset_path.

To run the facial recognition system run 'wearmask.py'

```
python3 wearmask.py
```
### The results were obtained as follows

![WhatsApp Image 2022-10-13 at 22 51 28](https://user-images.githubusercontent.com/88647994/195776224-79f5db64-95e7-43ac-9c9b-5b1164d1c5c8.jpeg)
![WhatsApp Image 2022-10-14 at 11 43 19](https://user-images.githubusercontent.com/88647994/195776388-8bba082d-7db4-416e-a777-aa73774429fd.jpeg)


#### Note

- The variable 'alternate = False' would put mask on every eight image and 'alternate = True' would put mask on alternate images.
- The variable 'xor = True/False' is for alternating the masking sequence.
- If you wish to obtain a dataset with 50% masked images, edit line 241 as 'xor = not xor'.

[![Anurag's GitHub stats](https://github-readme-stats.vercel.app/api?username=Opollo11)](https://github.com/anuraghazra/github-readme-stats)
