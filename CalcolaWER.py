import jiwer
import re 
def calculate_WER(transcription,transcriptionFromModel):
    
    #per calcolare il WER va rimossa la punteggiatura
    transcription=re.sub(r'[^a-zA-Z0-9\s]', '', transcription)
    transcriptionFromModel=re.sub(r'[^a-zA-Z0-9\s]', '', transcriptionFromModel)

    wer=jiwer.wer(transcription,transcriptionFromModel)

    print("WER:", wer )    
    return wer 

def accuracyFromWER(wer):
    return 1-wer



