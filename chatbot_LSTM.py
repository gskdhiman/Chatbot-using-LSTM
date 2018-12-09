import os
import json
import pandas as pd
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.recurrent import LSTM
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

word_emb_model_path = os.path.join('..','Google_News_vectors','GoogleNews-vectors-negative300.bin')

dataset_location = os.path.join('dataset','convAI2','export_2018-07-04_train.json')
dataset_location2 = os.path.join('dataset','convAI2','export_2018-07-05_train.json')

def data_from_json(file,list_of_keys):
    results = [] 
    def decode_each_dict(a_dict):
        try: 
            for k in list_of_keys:
                if k in a_dict:
                    results.append(str(a_dict[k]))
        except KeyError: 
            pass
        return a_dict
    with open(file) as fileobj:    
        json.loads(fileobj.read(), object_hook=decode_each_dict)  # Return value ignored.
    return results

data = data_from_json(dataset_location,['text'])
data2 = data_from_json(dataset_location2,['text'])

question =[v for i,v in enumerate(data) if i%2==0]
answer = [v for i,v in enumerate(data) if i%2!=0]

question2=  [v for i,v in enumerate(data2) if i%2==0]
answer2 = [v for i,v in enumerate(data2) if i%2!=0]

question.extend(question2)
answer.extend(answer2)
#hack if len mismatch
question = question[:len(question)-1]

df = pd.DataFrame({0:question,
                   1:answer
                   })
df.apply(lambda x: x.astype(str).str.lower())
df = df.drop_duplicates()  

df.info()  

ques_df = df.iloc[:,0]
ans_df = df.iloc[:,1]
#del df,question,data,data2,question2,answer,answer2,dataset_location,dataset_location2
mod = KeyedVectors.load_word2vec_format(word_emb_model_path, binary=True)
print('model loaded')
    
emb_dim = 300
padding_len = 15

def get_emb(element):
    emb=[]
    for word in element.split():
        try:
            emb.append(mod.wv[word])    
        except:
            emb.append(np.ones(emb_dim))    
    
    return pad_sequences([emb],maxlen=padding_len,dtype='float64')[0]

ques= ques_df.apply(lambda x : get_emb(x).astype('float64'))
ans= ans_df.apply(lambda x : get_emb(x).astype('float64'))
ques = np.array(ques.tolist())
ans = np.array(ans.tolist())

inp = Input(shape=(padding_len,emb_dim))
out = LSTM(300,return_sequences = True,activation='sigmoid',go_backwards = True,kernel_initializer = 'glorot_normal', recurrent_initializer='glorot_normal' )(inp)
out = LSTM(600,return_sequences = True,activation='sigmoid',go_backwards = True,kernel_initializer = 'glorot_normal', recurrent_initializer='glorot_normal')(out)
out = LSTM(300,return_sequences = True,activation='sigmoid',kernel_initializer = 'glorot_normal', recurrent_initializer='glorot_normal')(out)
model = Model(inputs=inp, outputs=out)

model.compile(optimizer = 'adam', loss= 'cosine_proximity', metrics=['accuracy'])
model.fit(ques,ans,epochs=200, verbose=1,validation_split=0.1,batch_size=4)
model.save('chatbot.h5')
#predictions = model.predict(test_ans)

model=load_model('chatbot.h5')
mod = KeyedVectors.load_word2vec_format(word_emb_model_path, binary=True)
while(True):
    input_query=input("Type your query/exit() to quit: ");
    if input_query =='exit()':
        break
    input_data = get_emb(input_query)
    vec_input = np.array(input_data.tolist()).reshape(1,padding_len,emb_dim)
    pred = model.predict(vec_input)
    output=[mod.most_similar([pred[0][i]])[0][0] for i in range(padding_len)]
    result = ' '.join(output)
    print(result)


