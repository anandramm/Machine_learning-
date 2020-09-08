import spacy
from spacy import displacy
nlp=spacy.load('en_core_web_sm')
doc2= nlp('Books are knowledge that is as deep as Pacific ocean. Iam going to read the  U.N magazine')
for token in doc2:
  print(token)  #Gives tokenized sentence
  print(token.label_)  #GIves the label
  print(token.pos_) #Gives the parts of Speech for the sentence
  print("\n")

#To get the graphical representation of the Parts of Speech we use despacy
displacy.render(doc2,style='dep',jupyter=True,options={'distance':100})