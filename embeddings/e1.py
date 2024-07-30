# natasha
import os 
from navec import Navec
from razdel import tokenize
import numpy as np

if not os.path.exists('../embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar'):
    os.system('wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar')


class e1_embedding:
    def __init__(self, ckpt = '../embeddings/navec_hudlit_v1_12B_500K_300d_100q.tar'):
        self.navec = Navec.load(ckpt)
        self.embedding_size = 300

    def apply_tokenizer(self, sentence):
        return [_.text for _ in tokenize(sentence)]
    
    def apply_navec(self,tokens):
        o = np.zeros(shape=(self.embedding_size, len(tokens)),dtype=np.float32)
        for i in range(len(tokens)):
            token = tokens[i]
            if self.navec.vocab.get(token, self.navec.vocab.unk_id) == self.navec.vocab.unk_id:
                o[:,i] = self.navec['<unk>']
            else:
                o[:,i] = self.navec[token]
        return o
    
    def __call__(self, batch):
        
        '''
        Bx1
        batch:
        [
            long sentnce 1,
            long sentence 2,
            ...
        ]
        '''
        if len(batch)==1 or type(batch)==str:
            if (type(batch)==list and len(batch)==1):
                x = batch[0]
            elif type(batch)==str:
                x = batch
            return self.apply_navec(self.apply_tokenizer(x))
        elif len(batch) > 1 and type(batch) != str:
            return [
                self.apply_navec(self.apply_tokenizer(el)) for el in batch
            ]