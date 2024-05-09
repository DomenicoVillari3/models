import os
from datasets import load_dataset, Audio
import numpy as np 
import jiwer
import time
import matplotlib.pyplot as plt
from pydub import AudioSegment

'''./main --model /home/domenico/whisper.cpp/models/ggml-base.bin
--file samples/jfk.wav --language en --output-txt 
--output-file ./samples/f.txtwhisper_init_from_file_with_params_no_state: 
'''

#Path to whisper dirercory
target_directory = '/home/domenico/whisper.cpp'
model_directory = '/home/domenico/whisper.cpp/models/ggml-base.bin'

os.chdir(target_directory)


def loadDataset():
    ds = load_dataset("google/fleurs", "it_it",trust_remote_code=True) #carico il dataset 
    ds = ds.cast_column("audio", Audio(sampling_rate=16000)) 
    return ds


def main():
    wer_list = []
    time_list=[]
    

    dataSet=loadDataset()


if __name__=="__main__":
    main()
