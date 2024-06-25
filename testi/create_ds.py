import re 

dataset=[]

with open("testi/EdipoRe.txt", "r") as f:
    with open("testi/EdipoRe2.txt", "w") as fw:
        edipo = f.read()
        
        #Rimuove i nomi dei personaggi
        edipo = re.sub(r'\b(EDIPO|SACERDOTE|CREONTE|CORO|TIRESIA|GIOCASTA|NUNZIO|SERVO|NUNZIO I|CORIFEO|PARODO)\b.', '\n', edipo)
        edipo = re.sub(r'\[.*?\]', '', edipo) #levo le quadre 
        edipo = re.sub(r'Sofocle\s+\d+\s+Edipo\s+Re', '', edipo)#elimina le intestazioni
        edipo = re.sub(r'\b[A-Z]+\b', '', edipo) #elimina tutte le parole in caps lock
        edipo=re.sub(r'\n\s*\n', '\n', edipo) #elimina le righe vuote
        fw.writelines(edipo)

with open("testi/EdipoRe2.txt", "r") as f:
    with open("testi/EdipoDS.txt", "w") as fw:
        for line in f:
            dataset.append(line.strip())
            print (line.strip())
            fw.writelines(line.strip()+"\n")
            

        print(len(dataset))
    

            