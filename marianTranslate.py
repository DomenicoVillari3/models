from transformers import AutoTokenizer, MarianMTModel
from datasets import load_dataset
import sacrebleu
import os 
import time
import csv
import numpy as np
import random


def to_csv(filename,run,bleu,avg_time):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['esecuzione', 'bleu_score','avg_time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            #scrivo riga
            writer.writerow({'esecuzione': run, 'bleu_score': bleu, 'avg_time': avg_time})

# Funzione di ordinamento basata sulla lunghezza della frase in inglese
def sort_by_sentence_length(item):
    return len(item['sentence_ita_Latn'])

def load_model(model_name):
    
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

def translate(model,tokenizer,sample_text):
    
    batch = tokenizer([sample_text], return_tensors="pt")
    generated_ids = model.generate(**batch)
    translated_text_from_text=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return translated_text_from_text


def main():
    src = "it"  # source language
    trg = "en"  # target language
    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

    model,processor=load_model(model_name)

    dataset = load_dataset("facebook/flores","ita_Latn-eng_Latn",split="devtest",trust_remote_code=True)
    dataset=sorted(dataset, key=sort_by_sentence_length)

   

    ita="sentence_ita_Latn"
    eng="sentence_eng_Latn"

    for i in range(len(dataset)):
        print(len(dataset[i][ita]))
    #print(dataset)
    references = []
    hypotheses = []
    t=[]

    for run in range(0):
        references.clear()
        hypotheses.clear()
        t.clear()

        #500 entry
        for i in range(len(dataset)//2):

            random_index = random.randint(0, len(dataset) - 1)
            print("traduzione {} su run {}\n".format(i,run))
            src_text=dataset[random_index][ita]
            dst_text=dataset[random_index][eng]
            init=time.time()
            translated_text = translate(model, processor, src_text)
            end=time.time()

            t.append(end-init)
            references.append(dst_text)
            hypotheses.append(translated_text)
            if i==0:
                print("traduzione {}".format(translated_text))

        references = [[ref] for ref in references]
        bleu = sacrebleu.corpus_bleu(hypotheses, references)
        print("run {}, BLEU score:{} \n\n".format(run,bleu.score))
        to_csv("marianTranlaste.csv",run,bleu.score,np.mean(t))



if __name__ == "__main__":
    main()