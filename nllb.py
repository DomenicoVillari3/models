# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import sacrebleu
import time
import numpy as np
from WriteMeanToCSV import writeBleu


#CARICO IL MODELLO CON LINGUAGGIO SORGENTE
def load_model(model_name,src): 
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=src)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return model,tokenizer

#TRADUZIONE
def translate(model,tokenizer,sample_text):
    batch = tokenizer(sample_text, return_tensors="pt")
    generated_ids = model.generate(**batch)
    translated_text_from_text=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return translated_text_from_text


def main():
    src = "ita_Latn"  # source language
    trg = "eng_Latn"  # target language
    model_name = "facebook/nllb-200-distilled-600M"

    #Carico il modello e il tokenizer usando la funzione load_model()
    model,tokenizer=load_model(model_name,src)
    #SETTO I LINGUAGGI
    tokenizer.src_lang=src
    tokenizer.tgt_lang=trg

    #carico il dataset
    dataset = load_dataset("facebook/flores","ita_Latn-eng_Latn",split="devtest",trust_remote_code=True)
    ita="sentence_ita_Latn"
    eng="sentence_eng_Latn"

    #print(dataset)
    references = []
    hypotheses = []
    t=[]

    #INFERENZA
    for i in range(len(dataset)):
        print("traduzione {}\n".format(i))
        src_text=dataset[i][ita]
        dst_text=dataset[i][eng]
        init=time.time()
        translated_text = translate(model, tokenizer, src_text)
        end=time.time()

        t.append(end-init)
        references.append(dst_text)
        hypotheses.append(translated_text)
        if i==0:
            print("traduzione {}".format(translated_text))

    #CALCOLO BLEU SCORE
    references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print("BLEU score:{} \n\n".format(bleu.score))


    writeBleu(filename="translate_means.csv",model=model_name,bleu=bleu.score,avg_time=np.mean(t))





if __name__ == "__main__":
    main()