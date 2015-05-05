import pickle
import json
import re
import sys
from nltk import pos_tag
from nltk.tokenize import RegexpTokenizer

def loadClassifier(modelfile):
    f = open(modelfile, 'rb')
    classifier = pickle.load(f)
    f.close()
    return classifier

def loadConfusionSet(confsetfile):
    confusionsetfile = open(confsetfile)
    confusionset = json.load(confusionsetfile)
    return confusionset

def formatline(splits, repl):
    i = 0
    newline = ''
    for i in range(len(repl)):
        split = splits[i]
        newline = newline + split + repl[i]
        i=i+1
    while (i < len(splits)):
        newline = newline + splits[i]
        i = i+1
    return newline

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

if __name__ == '__main__':
    classifier      = loadClassifier('model.pickle')
    confusionset    = loadConfusionSet('confusionset.json')
    confusionsetReg = ("|").join('\\b' + key + '(?!\w+)' for key in confusionset.keys())
    wordtokenizer   = RegexpTokenizer(confusionsetReg +'|'+ '\w+|[^\w\s]+')

    for line in sys.stdin.readlines():
        splits = re.split(confusionsetReg, line)
        if(len(splits) > 1):
            wordtokens    = wordtokenizer.tokenize(line)
            wordpostokens = postagtokens(wordtokens)
            repl = []
            for j in range(len(wordpostokens)):
                token = wordpostokens[j][0]
                if(token in confusionset.keys()):
                    f = generateFeatureSet(wordpostokens, j)
                    y1 = token
                    y2 = confusionset[y1]
                    prob_dist = classifier.prob_classify(f)
                    if(prob_dist.prob(y2) > prob_dist.prob(y1)):
                        repl.append(y2)
                    else:
                        repl.append(y1)
            newline = formatline(splits, repl)
            print newline.rstrip('\n')
        else:
            print line.rstrip('\n')
