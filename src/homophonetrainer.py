import json
import pickle
import re
from nltk.tokenize import RegexpTokenizer, PunktSentenceTokenizer
from nltk.corpus import abc, gutenberg, brown, reuters
from nltk.classify import maxent
from nltk import pos_tag

def loadConfusionSet(confsetfile):
    confusionsetfile = open(confsetfile)
    confusionset = json.load(confusionsetfile)
    return confusionset

def postagtokens(wordtokens):
    wordpostokens = pos_tag(wordtokens)
    wordpostokens.insert(0,('_BEG_','_BEG_'))
    wordpostokens.append(('_END_','_END_'))
    wordpostokens.append(('_END_','_END_'))
    return wordpostokens

def generateFeatureSet(wordpostokens, j):
    f = {}
    f["L1:" + wordpostokens[j-1][0].lower()] = 1
    f["L2:" + wordpostokens[j-2][0].lower()] = 1
    f["R1:" + wordpostokens[j+1][0].lower()] = 1
    f["R2:" + wordpostokens[j+2][0].lower()] = 1

    f["L1P:" + wordpostokens[j-1][1].upper()] = 1
    f["L2P:" + wordpostokens[j-2][1].upper()] = 1
    f["R1P:" + wordpostokens[j+1][1].upper()] = 1
    f["R2P:" + wordpostokens[j+2][1].upper()] = 1
    return f

def saveClassifierModel(classifier, modelfile):
    outputfile = open(modelfile,'wb')
    pickle.dump(classifier, outputfile, -1)
    outputfile.close()

def addSentencesToTrainingSet(trainingset, sentences):
    confusionset = loadConfusionSet('confusionset.json')
    confusionsetOr = ("|").join('\\b' + key + '(?!\w+)' for key in confusionset.keys())
    wordtokenizer = RegexpTokenizer(confusionsetOr +'|'+ '\w+|[^\w\s]+')

    for sentence in sentences:
        match = re.search(confusionsetOr, sentence)
        if(match):
            wordtokens     = wordtokenizer.tokenize(sentence)
            wordpostokens  = postagtokens(wordtokens)
            for j in range(len(wordpostokens)):
                token = wordpostokens[j][0]
                if(token in confusionset.keys()):
                    f = generateFeatureSet(wordpostokens, j)
                    y = token
                    trainingset.append((f,y))

def addWordPosTagsToTrainingSet(trainingset, postaggedsents):
    confusionset = loadConfusionSet('confusionset.json')
    for wordpostokens in postaggedsents:
        wordpostokens.insert(0,('_BEG_','_BEG_'))
        wordpostokens.append(('_END_','_END_'))
        wordpostokens.append(('_END_','_END_'))
        for j in range(len(wordpostokens)):
            token = wordpostokens[j][0]
            if(token in confusionset.keys()):
                f = generateFeatureSet(wordpostokens, j)
                y = token
                trainingset.append((f,y))

def addWikiDumpstoTrainingSet(trainingset, numfiles):
    for i in range(numfiles):
        inputfile = open('wiki'+str(i), 'r')
        for line in inputfile:
            line = line.decode('iso-8859-1').strip()
            sentences = sentencetokenizer.tokenize(line)
            addSentencesToTrainingSet(trainingset, sentences)
        inputfile.close()
        print("Added WikiDump " + str(i+1) + " to training set.")

if __name__ == '__main__':

    sentencetokenizer = PunktSentenceTokenizer()
    trainingset = []

    # WIKI DUMPS
    addWikiDumpstoTrainingSet(trainingset, 5)

    # BROWN CORPUS
    addWordPosTagsToTrainingSet(trainingset, brown.tagged_sents())

    # ABC CORPUS
    sentences = sentencetokenizer.tokenize(abc.raw())
    addSentencesToTrainingSet(trainingset, sentences)


    # REUTERS CORPUS
    # sentences = sentencetokenizer.tokenize(reuters.raw())
    # addSentencesToTrainingSet(trainingset, sentences)

    # GUTENBERG CORPUS
    # sentences = sentencetokenizer.tokenize(gutenberg.raw())
    # addSentencesToTrainingSet(trainingset, sentences)

    classifier = maxent.MaxentClassifier.train(trainingset, 'MEGAM', max_iter=1000)
    saveClassifierModel(classifier, 'model.pickle')
