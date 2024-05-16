import torchaudio
import numpy as np
import torch
from transformers import SeamlessM4Tv2Model, AutoProcessor
from datasets import load_dataset
import time
from CalcolaWER import calculate_WER,accuracyFromWER
from Myplot import my_plot
from WriteMeanToCSV import WriteMeanToCSV,WriteValues

wer_list=[]
t_list=[]
acc_list=[]

def LoadModel():
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
    return model,processor

def use_seamless(model,processor,audio_inputs):
    # generate translation
    start=time.time()
    output_tokens = model.generate(**audio_inputs, tgt_lang="ita", generate_speech=False)
    translated_text_from_audio = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    end=time.time()
    elapsed = end-start

    return translated_text_from_audio,elapsed


def main():
    model,processor=LoadModel()

    for run in range(20):
        wer_list = []
        time_list=[]
        acc_list = []
        #CARICAMENTO DEL DATASET
        dataset = load_dataset("google/fleurs", "it_it", split="test", streaming=True,trust_remote_code=True)

        for audio_sample in dataset:
            transcription=audio_sample["transcription"]
            audio_sample = audio_sample["audio"]
            audio = torch.tensor(audio_sample["array"])
            #print(f"Sampling rate: {audio_sample['sampling_rate']}")
            audio = torchaudio.functional.resample(audio, orig_freq=audio_sample['sampling_rate'], new_freq=model.config.sampling_rate)

            # process input
            audio_inputs = processor(audios=audio, return_tensors="pt")


            ipotesi,t=use_seamless(model,processor,audio_inputs)
    

            wer=calculate_WER(transcription,ipotesi)
            acc=accuracyFromWER(wer)

            wer_list.append(wer)
            t_list.append(t)
            acc_list.append(acc)
            print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))

        #TRASCRIZIONI SU FILE CSV DEI VALORI MEDI 
        WriteMeanToCSV("whisperMEAN.csv",run,avg_wer=np.mean(wer_list),avg_time=np.mean(time_list),avg_accuracy=np.mean(acc_list)) 
        WriteValues("whisper.csv",run,wer_l=wer_list,time_l=time_list,accuracy_l=acc_list)     

    my_plot("SM4T.csv","SM4T.csv")

    


if __name__ == "__main__":
    main()