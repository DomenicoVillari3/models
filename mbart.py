from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from datasets import load_dataset
import sacrebleu 
import time
import numpy as np
from WriteMeanToCSV import writeBleu


#CARICO IL MODELLO 
def load_model(model_name):
    model = MBartForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MBart50TokenizerFast.from_pretrained(model_name)
    return model,tokenizer

#TRADUZIONE
def translate(model,tokenizer,sample_text,tgt):    
    batch = tokenizer([sample_text], return_tensors="pt")
    generated_ids = model.generate(**batch,forced_bos_token_id=tokenizer.lang_code_to_id[tgt])
    translated_text_from_text=tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return translated_text_from_text


def main():
    src = "it_IT"  # source language
    trg = "en_XX"  # target language
    model_name ="facebook/mbart-large-50-many-to-many-mmt"
    model,tokenizer=load_model(model_name)
    tokenizer.src_lang=src

    #load dataset
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
        translated_text = translate(model, tokenizer, src_text,trg)
        end=time.time()

        t.append(end-init)
        references.append(dst_text)
        hypotheses.append(translated_text)
        if i==0:
            print("traduzione {}".format(translated_text))

    references = [[ref] for ref in references]
    bleu = sacrebleu.corpus_bleu(hypotheses, references)
    print("BLEU score:{} \n\n".format(bleu.score))
    writeBleu(filename="translate_means.csv",model=model_name,bleu=bleu.score,avg_time=np.mean(t))





if __name__ == "__main__":
    main()