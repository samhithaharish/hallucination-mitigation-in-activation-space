"""Importable train-only encoders, signed response models, and correctness controls."""
import re
import numpy as np
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.random_projection import SparseRandomProjection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPRegressor
from .data import SEED

from experiments.paths import SEMANTIC_PATH


def semantic_view(text):
    # MiniLM has a short input window: put the question and answer before evidence.
    question = text.rsplit('Question: ',1)[-1].split('\n',1)[0]
    answer = text.rsplit('Initial answer: ',1)[-1]
    evidence = text.split('Question: ',1)[0]
    return 'Question: '+question+'\nInitial answer: '+answer+'\nEvidence: '+evidence


def semantic_embeddings(texts):
    from sentence_transformers import SentenceTransformer
    model=SentenceTransformer(SEMANTIC_PATH,device='cpu',local_files_only=True)
    return model.encode([semantic_view(t) for t in texts],batch_size=128,normalize_embeddings=True,
                        show_progress_bar=False,convert_to_numpy=True).astype(np.float32)


class Encoder:
    def __init__(self,components=64,text_components=256,projection_components=512,semantic=True):
        self.components=components;self.text_components=text_components
        self.projection_components=projection_components;self.semantic=semantic

    def fit(self,states,metadata,texts):
        self.pcas=[];self.time_pcas=[]
        for layer in range(states.shape[1]):
            x=states[:,layer].reshape(len(states),-1).astype(np.float32)
            self.pcas.append(PCA(min(self.components,*x.shape),svd_solver='randomized',random_state=SEED).fit(x))
            self.time_pcas.append([PCA(min(32,len(states),states.shape[-1]),svd_solver='randomized',random_state=SEED).fit(
                states[:,layer,t].astype(np.float32)) for t in range(states.shape[2])])
        self.projection=SparseRandomProjection(n_components=min(self.projection_components,int(np.prod(states.shape[1:]))),
            dense_output=True,random_state=SEED).fit(states.reshape(len(states),-1))
        self.vectorizer=TfidfVectorizer(max_features=60000,min_df=2,ngram_range=(1,2),sublinear_tf=True,strip_accents='unicode')
        sparse=self.vectorizer.fit_transform(texts)
        self.svd=TruncatedSVD(min(self.text_components,sparse.shape[0]-1,sparse.shape[1]-1),random_state=SEED).fit(sparse)
        return self

    def transform(self,states,metadata,texts,semantic=None):
        layers=[p.transform(states[:,i].reshape(len(states),-1).astype(np.float32)) for i,p in enumerate(self.pcas)]
        activation=np.concatenate(layers,axis=1)
        low=np.concatenate([l[:,:16] for l in layers],axis=1)
        projected=self.projection.transform(states.reshape(len(states),-1).astype(np.float32))
        lexical=self.svd.transform(self.vectorizer.transform(texts)).astype(np.float32)
        confidence=np.asarray(metadata,dtype=np.float32)
        text=np.concatenate([confidence,lexical],axis=1)
        if self.semantic:
            sem=semantic_embeddings(texts) if semantic is None else semantic
            behavior=np.concatenate([text,sem],axis=1)
        else:behavior=text
        families={'confidence':confidence,'lexical':text,'semantic':behavior,'activation':activation,
            'full':np.concatenate([behavior,activation],axis=1),'random':projected,
            'full_random':np.concatenate([behavior,projected],axis=1),
            'full_pca16':np.concatenate([behavior,low],axis=1)}
        for i,l in enumerate(layers):families[f'layer{i}']=np.concatenate([behavior,l],axis=1)
        for t,name in enumerate(('prompt','answer')):
            components=[ps[t].transform(states[:,i,t].astype(np.float32)) for i,ps in enumerate(self.time_pcas)]
            families[name]=np.concatenate([behavior,*components],axis=1)
        return {k:np.asarray(v,dtype=np.float32) for k,v in families.items()}


class GainModel:
    def __init__(self,kind='ridge',alpha=10.,seed=0):
        self.kind=kind;self.alpha=alpha;self.seed=seed

    def fit(self,x,y):
        self.scaler=StandardScaler().fit(x)
        if self.kind=='ridge':self.model=Ridge(alpha=self.alpha)
        elif self.kind=='mlp':
            self.model=MLPRegressor(hidden_layer_sizes=(128,),alpha=self.alpha,batch_size=128,max_iter=300,
                early_stopping=True,validation_fraction=.15,n_iter_no_change=20,random_state=self.seed)
        else:raise ValueError(self.kind)
        self.model.fit(self.scaler.transform(x),y)
        return self

    def predict(self,x):
        gains=self.model.predict(self.scaler.transform(x)).clip(-1,1)
        return np.column_stack([np.zeros(len(x)),gains])


class CorrectnessGate:
    def __init__(self,c=1.):self.c=c

    def fit(self,x,f1,em):
        correct=em[:,0]==1;gains=f1-f1[:,:1]
        self.scaler=StandardScaler().fit(x)
        self.probability=float(correct.mean());self.classifier=None
        if len(np.unique(correct))==2:
            self.classifier=LogisticRegression(C=self.c,max_iter=2000).fit(self.scaler.transform(x),correct)
        self.gain_correct=gains[correct].mean(0) if correct.any() else gains.mean(0)
        self.gain_incorrect=gains[~correct].mean(0) if (~correct).any() else gains.mean(0)
        return self

    def predict(self,x):
        p=self.classifier.predict_proba(self.scaler.transform(x))[:,1:2] if self.classifier is not None else self.probability
        gains=p*self.gain_correct+(1-p)*self.gain_incorrect
        return np.broadcast_to(gains,(len(x),len(self.gain_correct))).copy()
