import spacy
import neuralcoref


nlp = spacy.load('en_core_web_sm')
neuralcoref.add_to_pipe(nlp, conv_dict={'Deepika': ['woman', 'actress']})
doc = nlp(u'Deepika has a dog. She loves him. The movie star has always been fond of animals')
print(doc._.coref_resolved)
