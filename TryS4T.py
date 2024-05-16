import torchaudio
import numpy as np
import torch
from transformers import SeamlessM4Tv2Model, AutoProcessor
import librosa
from datasets import load_dataset
import time
from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

wer_list=[]
t_list=[]
acc_list=[]

# let's load an audio sample from an Hindi speech corpus
dataset = load_dataset("google/fleurs", "it_it", split="test", streaming=True)
for audio_sample in dataset:
    transcription=audio_sample["transcription"]
    audio_sample = audio_sample["audio"]
    audio = torch.tensor(audio_sample["array"])
    #print(f"Sampling rate: {audio_sample['sampling_rate']}")
    audio = torchaudio.functional.resample(audio, orig_freq=audio_sample['sampling_rate'], new_freq=model.config.sampling_rate)

    #audio, orig_freq =  torchaudio.load("C:\\Users\\domen\\Desktop\\Test_models\\models\\test.wav")
    #audio =  torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16_000) # must be a 16 kHz waveform array

    # process input
    audio_inputs = processor(audios=audio, return_tensors="pt")

    # generate translation
    start=time.time()
    output_tokens = model.generate(**audio_inputs, tgt_lang="ita", generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    end=time.time()
    
    elapsed = end-start
    wer=calculate_WER(transcription,translated_text_from_audio)
    acc=accuracyFromWER(wer)

    wer_list.append(wer)
    t_list.append(elapsed)
    acc_list.append(acc)

    my_plot(wer_list,t_list,"SM4T")

    print(f"Translation from audio: {translated_text_from_audio}")