from transformers import AutoTokenizer, MarianMTModel
from datasets import load_dataset
import sacrebleu
import time
import numpy as np
from WriteMeanToCSV import writeBleu


#CARICO IL MODELLO
def load_model(model_name): 
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model,tokenizer

#TRADUZIONE
def translate(model,tokenizer,sample_text):
    #tokenize del testo in input ("pythorch")
    batch = tokenizer([sample_text], return_tensors="pt")
    #genera la traduzione ("hello world") usando il modello caricato e il tokenizer
    generated_ids = model.generate(**batch)
    #decodifica la traduzione in testo ("hello world") usando il tokenizer
    translated_text_from_text=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return translated_text_from_text


def main():
    src = "it"  # source language
    trg = "en"  # target language
    model_name = f"Helsinki-NLP/opus-mt-{src}-{trg}"

    #Carico il modello e il tokenizer usando la funzione load_model()
    model,tokenizer=load_model(model_name)

    #Carico dataset
    dataset = load_dataset("facebook/flores","ita_Latn-eng_Latn",split="devtest",trust_remote_code=True)
    ita="sentence_ita_Latn"
    eng="sentence_eng_Latn"


    #print(dataset)
    references = []
    hypotheses = []
    t=[]

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

    #Calcolo e stampo BLEU score
    references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print("BLEU score:{} \n\n".format(bleu.score))

    writeBleu(filename="translate_means.csv",model="Helsinki-NLP/opus-mt-it-en",bleu=bleu.score,avg_time=np.mean(t))




if __name__ == "__main__":
    main()