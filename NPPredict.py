from nltk import word_tokenize, pos_tag
from NPLearner import default_feature_func

try:
    import cPickle as pickle
except:
    import pickle

IOB_LABEL_MAP = {0: "O", 1: "B-NP", 2:"I-NP"}
IO_LABEL_MAP = {0: "O", 1: "I-NP"}


#TODO: fix mdoel location
IO_MODEL_LOC = "models/IO_labeling_models/nltkMaximumEntropyClass_default.pkl"
IOB_MODEL_LOC = "models/IOB_labeling_models/nltkMaximumEntropyClass_default.pkl"

#TODO: implement function to take in a string and return an array of labels for every single word

class NPPredictor:
    def __init__(self, model, feature_func, label_map, model_type="NLTK"):

        self.model = model
        self.feat_func = feature_func
        self.label_map = label_map

    def predict_sentence(self, sentence, verbose=False):
        """ Predicts the labels for every word and punctuation
        Inputs:
            sentence (string): input sentence

        Returns:
            labels (list): list containing the labels for every tokenized word.
        
        """
        
        # tokenize and then attach POS labeling
        sentence = self.tokenize_pos(sentence)
        sentence_feats = self.feat_func([sentence])
        
        predictions = self.model.predict(sentence_feats)
        
        if verbose:
            print(self.convert_to_str_labels(predictions, sentence))

        return predictions

    def tokenize_pos(self, sentence):
        """ Converting a string to a tuple of tuples:
        Inputs:
            sentence (string): input sentence.

        Returns:
            pos (tuple): Each element is a tuple, with the first element within as the 
                word and the second element as the POS tag.
        
        """

        assert type(sentence) == str, "Input type has to be string"

        tokenized = word_tokenize(sentence)
        pos_tagged = pos_tag(tokenized)
        
        return pos_tagged

    
    def predict_paragraph(self, paragraph):
        """ Predicts the labels for every word in the paragraph by breaking it
        down into sentences and calling predict_sentence.

        Inputs:
            paragraph (string): input paragraph
        
        Returns:
            labels (list): list containing the labels for every tokenized word.
        
        """
        # TODO: break paragraph into sentences.
        
        pass

    def convert_to_str_labels(self, output, sent_tokenized):
        """ Converting the numerical outputs of the model into human understandable labels.
        Input:
            output (list): list of numerical labels.
            sent_tokenized (list): list of tokens in sentences.
        Output:
            str_labels (list): list of str labels.
        """
        
        return [(sent_tokenized[i][0], self.label_map[label]) for (i, label) in enumerate(output)]



def main():
    # TODO: read model from .pkl file
    IO_model = pickle.load(open(IO_MODEL_LOC, "rb"))
    IOB_model = pickle.load(open(IOB_MODEL_LOC, "rb"))
    feat_func = default_feature_func
    
    input_sentence = "The blue cat is sitting on the red mat in the grey house."
    
    IO_npp = NPPredictor(IO_model, feat_func, IO_LABEL_MAP)
    print(IO_npp.predict_sentence(input_sentence, verbose=True))

    IOB_npp = NPPredictor(IOB_model, feat_func, IOB_LABEL_MAP)
    print(IOB_npp.predict_sentence(input_sentence, verbose=True))

if __name__=="__main__":
    
    main()
    

