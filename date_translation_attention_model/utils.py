import numpy as np
from faker import Faker
from babel.dates import format_date
from typing import Dict
from keras.utils import to_categorical

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY'
           ]
LOCALES = ['en_US']

fake = Faker()


def load_random_date(): 
    dt = fake.date_object()
    human_readable = format_date(dt, format=np.random.choice(FORMATS), locale=np.random.choice(LOCALES))
    human_readable = human_readable.lower()
    human_readable = human_readable.replace(',', '')
    machine_readable = dt.isoformat()
    
    return human_readable, machine_readable, dt


def load_dataset(m: int): 
    """_summary_

    Args:
        m -- (int): number of training examples

    Returns:
        dataset -- list[(human, machine)]: a list of tuples (human readable date, machine readable date)
        human_vocab -- dict[str, int]: a python dictionary mapping app characters used in human readable date to integer-valued index.
        machine_vocab -- dict[str, int]: a python dic mapping app characters used in machine readable dates to an integer-valued index.
        inv_machine_vocab -- dict[int, str]: mapping indices to characters (reversed dict of machine_vocab)
    """
    dataset = []
    
    human = set()
    machine = set()
    
    for i in range(m): 
        h, m, _ = load_random_date()
        
        dataset.append((h, m))
        # Add characters into sets
        human.update(tuple(h))
        machine.update(tuple(m))
        
    # Sort characters alphabetically, then map chars to their indices
    human_vocab = dict(zip(sorted(human) + ['<unk>', '<pad>'], range(len(human) + 2)))
    
    inv_machine_vocab = dict(enumerate(sorted(machine)))
    machine_vocab = {w: i for i, w in inv_machine_vocab.items()}

    return dataset, human_vocab, machine_vocab, inv_machine_vocab


def string_to_int(string: str, length, vocab: Dict[str, int]): 
    """_summary_

    ## Args:
        string -- str: string to be converted to characters' indices
        length -- int: length for sequence
        vocab -- Dict[str, int]: a python dict mapping characters to their indices
    ## Returns: 
        indices -- List[int]: mapped from string to indices
    """
    
    string: str = string.lower().replace(',', '')
    if len(string) > length: 
        string = string[:length]
    
    indices: list[int] = list(map(lambda x: vocab.get(x, '<unk>'), string))
    if len(string) < length: 
        indices = indices + [vocab['<pad>']] * (length - len(indices))
    assert len(indices) == length, 'Wrong length for indices'
    return indices

def preprocessing(dataset, human_vocab, machine_vocab, Tx, Ty): 
    """_summary_

    ## Args:
        dataset -- list[(human, machine)]: a list of tuples (human readable date, machine readable date)
        human_vocab -- Dict[str, int]: a python dictionary mapping app characters used in human readable date to integer-valued index.
        machine_vocab -- Dict[str, int]: a python dic mapping app characters used  in machine readable dates to an integer-valued index.
        Tx -- (int): maximum of human readable date.
        Ty -- (int): maximum date lenght ("YYYY-MM-DD").
    ## Returns: 
        X -- ndrray[m, Tx]: a preprocessed version of the human readable dates in the training set.
        Y -- list(ndarray[Ty,]): ;a preprocessed version of machine readable dates in the training set.
        Xoh -- ndarray[m, Tx, len(n_human_vocab)]: a one-hot version of X
        Yoh -- ndarray[m, Ty, len(n_machine_vocab)]: a one-hot version of Y
    """
    X_: tuple[str]
    Y_: tuple[str]
    X_, Y_ = zip(*dataset)    
    
    # ndarray with shape (m, Tx)
    X: np.ndarray[m, Tx] = np.array([string_to_int(x, Tx, human_vocab) for x in X_])
    # list contains m * ndarrays of shape (Ty,)
    Y: list[np.ndarray[Ty]] = list(map(lambda y: string_to_int(y, Ty, machine_vocab), Y_))

    n_human_vocab = len(human_vocab)
    n_machine_vocab = len(machine_vocab)
    
    Xoh: np.ndarray[m, Tx, len(n_human_vocab)] = np.array(list(map(lambda x: to_categorical(x, num_classes=n_human_vocab) , X)))
    Yoh: np.ndarray[m, Ty, len(n_machine_vocab)] = np.array(list(map(lambda x: to_categorical(x, num_classes=n_machine_vocab), Y)))
    
    return X, Y_, Xoh, Yoh