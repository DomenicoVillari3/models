import jiwer
def calculate_WER(transcription,transcriptionFromModel):
    
    #per calcolare il WER va rimossa la punteggiatura
    transcription=transcription.lower().replace(".","").replace(",","")
    transcriptionFromModel=transcriptionFromModel.lower().replace(".","").replace(",","")

    wer=jiwer.wer(transcription,transcriptionFromModel)

    print("WER:", wer )    
    return wer 

def accuracyFromWER(wer):
        return 1-wer

