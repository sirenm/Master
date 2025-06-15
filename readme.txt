How to use this system:

- I have used anaconda for environments
- The packages I have used in this project is listed in requirements.txt

Create dataset:
- Datasets are already in github repo, but if it wanted to re-generate them, you need to download the Crema dataset from Kaggle (e.g. https://www.kaggle.com/datasets/dmitrybabko/speech-emotion-recognition-en)
- Then run "python3 Dataset.py"

Train a model:

- There are four files for training models
    - WavLMBasePlusTrain
    - WavLMLargeTrain
    - Wav2Vec2BaseTrain
    - Wav2Vec2LargeTrain

- These files mainly contain the same code, but are set up for the specific models. The ptm is loaded, the training and validation datasets are used,
and the model is saved within the corresponding name, for instance, WavLMBasePlus. These are hardcoded, and can be changed, but the scripts are ready to 
be run.

- Run the training scripts using "python3 XXX.py", for instance, "python3 WavLMLargeTrain.py"

Run a model:

- There are four files for testing models
    - WavLMBasePlusRun
    - WavLMLargeRun
    - Wav2Vec2BaseRun
    - Wav2Vec2LargeRun

- These files mainly contain the same code, but are set up for the specific models. The trainedmodel is loaded and  the test datasets are used.
These are hardcoded, and can be changed, but the scripts are ready to 
be run.

- Run the testing scripts using "python3 XXX.py", for instance, "python3 WavLMLargeRun.py"

- REMEMBER: If you want to run a model, you must train it first. If you want to run a WavLMLarge model, train it, and according to that code the model will be saved in a folder named WavLMLarge.
            If you enter the WavLMLargeRun file, the model that it currently loads is the WavLMLarge model.


Use data augmentation:

- There are four files for data augmentation
    - Dataset_AWGN (does one round of AWGN)
    - Dataset_PitchShifting (does one round of PitchShifting)
    - Dataset_Stretching (does one round of Time stretching)
    - Dataset_DA_all (does two rounds of AWGN, PitchShifting, Time stretching)

- When you run these files they generate new folders containing the data. 
  If you want to use this data for training, you must remember to change the dataset name in the Train-file.

- Run the data augmentation scripts using "python3 XXX.py", for instance, "python3 Dataset_AWGN.py"