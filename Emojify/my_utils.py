import numpy as np
import csv
import emoji

def read_glove_vec(filepath='./data/fake_glove.6B.50d.txt'): 
    word_to_vec = {}
    word_to_index = {}
    index_to_word = {}
    words = set()
    with open(filepath) as f:
        for line in f.readlines(): 
            # remove '\n' and split the sentence to list
            line = line.strip()
            line_to_list = line.split()
            
            word = line_to_list[0]
            words.add(word)
            word_to_vec[word] = np.array(line_to_list[1:], dtype=float)
            
    for i, w in enumerate(sorted(words)): 
        word_to_index[w] = i
        index_to_word[i] = w
        
    return word_to_vec, word_to_index, index_to_word

def softmax(x: np.ndarray): 
    """_summary_

    Args:
        x (ndarray, (n_classes, m)): input
    Returns: 
        a (ndarray, (n_classes, m)): applied stable softmax 
    """
    
    normalized = np.exp(x - np.max(x, axis=0, keepdims=True))
    sum = np.sum(normalized, axis=0)
    a = normalized/ sum
    return a

def xavier_init(n_out, n_in): 
    """_summary_

    Args:
        n_out (int): number of output's features
        n_in (int): number of input's features

    Returns:
        W (ndarray, (n_out, n_in)): Weights
        b (mdarray, (n_out, 1)): Biases
    """
    W = np.random.randn(n_out, n_in) * np.sqrt(1/n_in)
    b = np.zeros((n_out, 1))
    
    return W, b

def read_csv(filename='data/train_emoji.csv'): 
    
    with open(filename) as csvDataFile: 
        phrase = []
        emoji = []
        csvReader = csv.reader(csvDataFile)
        for row in csvReader: 
            phrase.append(row[0])
            emoji.append(row[1])
    X = np.array(phrase)
    Y = np.array(emoji)
    return X, Y


emoji_dictionary = {
    #"0": "\u2764\uFE0F",    # :heart: prints a black instead of red heart depending on the font
    "0": "❣️", 
    "1": ":baseball:",
    "2": "😂",
    "3": ":disappointed:",
    "3": "😞",
    "4": ":fork_and_knife:"
}

def label_to_emoji(label):
    """
    Converts a label (int or string) into the corresponding emoji code (string) ready to be printed
    """
    return emoji.emojize(emoji_dictionary[str(label)])
              
def print_predictions(X, pred, true_label=None, count=None):
    print()
    _count = 0
    for i in range(X.shape[0]):
        if type(true_label) == type(None): 
            print(X[i], label_to_emoji(int(pred[i])))
        else: print(f'Expected emoji: {label_to_emoji(int(true_label[i]))}|', X[i], label_to_emoji(int(pred[i])))
            
        
        if count != None: 
            
            if  _count < count: 
                _count += 1
            else: break