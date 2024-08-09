from pathlib import Path
import pkgutil
from datasets import load_dataset
import datasets
import numpy as np 
import time
import re
import csv
from jiwer import wer
try:
    import whisper_cpp_python
except FileNotFoundError:
    regex = r"(\"darwin\":\n\s*lib_ext = \")\.so(\")"
    subst = "\\1.dylib\\2"

    print("fixing and re-importing whisper_cpp_python...")
    # load whisper_cpp_python and substitute .so with .dylib for darwin
    package = pkgutil.get_loader("whisper_cpp_python")
    whisper_path = Path(package.path)
    whisper_cpp_py = whisper_path.parent.joinpath("whisper_cpp.py")
    content = whisper_cpp_py.read_text()
    result = re.sub(regex, subst, content, 0, re.MULTILINE)
    whisper_cpp_py.write_text(result)

    import whisper_cpp_python

def accuracyFromWER(wer):
    return 1 - wer

def clean(stringa):
    cleaned_string = re.sub(r'[^\w\s]', '', stringa)  # Rimuove le punteggiature e le virgolette
    cleaned_string = cleaned_string.lower()  # Converte tutto in minuscolo
    return cleaned_string

def use_model(model, audio):
    start = time.time()
    result = model.transcribe(audio, language="it")
    end = time.time() - start
    return result["text"], end

def main():
    wer_list = []
    time_list = []
    acc_list = []
    
    dataSet = load_dataset("google/fleurs", "it_it", trust_remote_code=True,split="test")  # Carico il dataset 
    
    with open('results_whisper_cpp.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Esecuzione","Entry", "Prediction", "WER", "Time"])
        model = whisper_cpp_python.Whisper("/home/domenico/whisper.cpp/models/ggml-large-v3.bin", n_threads=4)

        for i in range(1, 16):
            for j, entry in enumerate(dataSet):
                if j >= 300:
                    break
                transcription = entry["transcription"]
                path = entry["path"].split("/")
                path[-1] = entry["audio"]["path"]
                audio = ""
                audio = "/".join(path)
                print(audio)
                ipotesi, t = use_model(model, audio)
                ipotesi = clean(ipotesi)
                time_list.append(t)
                _wer = wer(transcription, ipotesi)
                wer_list.append(_wer)
                acc_list.append(accuracyFromWER(_wer))
                #writer.writerow([i, transcription, ipotesi, _wer, t])
                print(f"Ciclo: {i}, dato: {j}")
                print("\n ipotesi: {}\ntrascrizione: {} \n".format(ipotesi, transcription))
        

                

if __name__ == "__main__":
    main()
