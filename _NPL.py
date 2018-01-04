#####
# Playing around with NP chunker provided by NLTK 
#   - Pretty much exact copy of code explained in http://www.nltk.org/book/ch07.html that I wrote while reading through
#   - Instead of hardcoding grammars, this naive implementation uses the Maximum Entropy classifer as an attempt to learn the grammar rules
#####

import nltk
from nltk.corpus import conll2000 #Download corpus with: nltk.download('conll2000')

class ConsecutiveNPChunkTagger(nltk.TaggerI):

    def __init__(self, train_sents):
        '''
        Maps chunk trees in training corpus into tag sequences
        '''
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history) 
                train_set.append( (featureset, tag) )
                history.append(tag)
        print("Training....")

        # Error occurred when trying to download and use the Magem algorithm, using GIS instead.
        # Haven't yet looked into which algorithm means exactly what so I need to look at that later
        
        self.classifier = nltk.MaxentClassifier.train( # Maximum Entropy Classifier
            train_set, algorithm='GIS', trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        '''
        Converts tag sequence back into chunk tree representation
        '''
        print("Parsing...")
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)


def npchunk_features(sentence, i, history):
    '''
    Extracting various features to improve chunker performance
    '''
    word, pos = sentence[i]
    if i == 0:
         prevword, prevpos = "<START>", "<START>"
    else:
         prevword, prevpos = sentence[i-1]
    if i == len(sentence)-1:
         nextword, nextpos = "<END>", "<END>"
    else:
         nextword, nextpos = sentence[i+1]
    return {"pos": pos,
            "word": word,
            "prevpos": prevpos,
            "nextpos": nextpos,
            "prevpos+pos": "%s+%s" % (prevpos, pos), 
            "pos+nextpos": "%s+%s" % (pos, nextpos),
            "tags-since-dt": tags_since_dt(sentence, i)} 
             
def tags_since_dt(sentence, i):
    '''
    Output:
        String that describes the set of all part-of-speech tags encountered since beginning of sentence OR most recent determiner 'DT'
    '''
    tags = set()
    for word, pos in sentence[:i]:
        if pos == 'DT':
            tags = set()
        else:
            tags.add(pos)
    return '+'.join(sorted(tags))

def main():
    test_sents = conll2000.chunked_sents('test.txt', chunk_types=['NP'])
    train_sents = conll2000.chunked_sents('train.txt', chunk_types=['NP'])

    chunker = ConsecutiveNPChunker(train_sents)
    print(chunker.evaluate(test_sents))


if __name__ == "__main__":
    main()
