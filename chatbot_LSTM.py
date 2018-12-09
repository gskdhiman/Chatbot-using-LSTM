import os
import json
import pandas as pd
dataset_location = os.path.join('dataset','convAI2','export_2018-07-04_train.json')

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

question =[v for i,v in enumerate(data) if i%2==0]
answer = [v for i,v in enumerate(data) if i%2!=0]

df = pd.DataFrame({0:question,
                   1:answer
                   })
df = df.drop_duplicates()    
    
