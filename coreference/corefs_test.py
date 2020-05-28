import spacy
import neuralcoref


nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, conv_dict={'Deepika': ['woman', 'actress']})
doc = nlp("I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!")
print(doc._.coref_resolved)
