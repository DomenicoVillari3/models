#python 3.10.12
from transformers import SeamlessM4Tv2Model
from transformers import AutoProcessor
from datasets import load_dataset
import sacrebleu
import os 
import time
import csv
import numpy as np


def to_csv(filename,model,bleu,avg_time):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
            
            # Definisce i nomi delle colonne nel file CSV
            fieldnames = ['model', 'bleu_score','avg_time']
            writer = csv.DictWriter(file, fieldnames=fieldnames)

            #scrivo intestazione
            if not file_exists:
                writer.writeheader()
            
            #scrivo riga
            writer.writerow({'model': model, 'bleu_score': bleu, 'avg_time': avg_time})

def load_model(model_name):
    model = SeamlessM4Tv2Model.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    return model,processor

def translate(model,processor,txt,src="ita",tgt="eng"):
    text_inputs = processor(text = txt, src_lang=src, return_tensors="pt")
    output_tokens = model.generate(**text_inputs, tgt_lang=tgt, generate_speech=False)
    translated_text_from_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
    #print(f" Translation from text: {translated_text_from_text}")
    return translated_text_from_text


def main():
    model_name="facebook/seamless-m4t-v2-large"
    model,processor=load_model(model_name)

    dataset = load_dataset("facebook/flores","ita_Latn-eng_Latn",split="devtest",trust_remote_code=True)
    ita="sentence_ita_Latn"
    eng="sentence_eng_Latn"
    #print(dataset)
    references = []
    hypotheses = []
    t=[]

    for i in range(len(dataset)):
        print("traduzione {} \n".format(i))
        src_text=dataset[i][ita]
        dst_text=dataset[i][eng]
        init=time.time()
        translated_text = translate(model, processor, src_text, src="ita", tgt="eng")
        end=time.time()

        t.append(end-init)
        references.append(dst_text)
        hypotheses.append(translated_text)

        
        print("traduzione {}".format(translated_text))

    references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print("BLEU score:{} \n\n".format(bleu.score))


    to_csv(filename="translate_means.csv",model="sm4t_translate",bleu=bleu.score,avg_time=np.mean(t))



if __name__ == "__main__":
    main()