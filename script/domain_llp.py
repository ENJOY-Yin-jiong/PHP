import pandas as pd
import os
import json


domain = dict()

domain["human activities"] = ["Speech", "Cheering", "Singing", "Clapping", "Baby_laughter", "Baby_cry_infant_cry"]
domain["animal activities"] = ["Dog", "Cat", "Chicken_rooster"]
domain["music performances"] = ["Cello", "Violin_fiddle", 'Accordion', 'Banjo', 'Acoustic_guitar']
domain["vehicle sounds"] = ["Car", "Motorcycle", 'Helicopter', 'Chainsaw', 'Lawn_mower']
domain["domestic environments"] = ["Frying_(food)", "Basketball_bounce", "Vacuum_cleaner", "Fire_alarm", "Telephone_bell_ringing", "Blender"]


train_data = pd.read_csv('/opt/data/private/dataset/AVVP/LLP_dataset/labels/AVVP_train.csv', header=0, sep='\t')
test_data = pd.read_csv('/opt/data/private/dataset/AVVP/LLP_dataset/labels/AVVP_test_pd.csv', header=0, sep='\t')


for index, row in train_data.iterrows():
    sample_domain = []
    categories = row["event_labels"].split(',')
    for cat in categories:
        for key, value in domain.items():
            if cat in value:
                sample_domain.append(key)
                train_data.loc[index, ("Domain")] = key
                break
    
    if len(sample_domain) != len(categories):
        print('----------------------------------------')
        print(row)

for index, row in test_data.iterrows():
    sample_domain = []
    categories = row["event_labels"].split(',')
    for cat in categories:
        for key, value in domain.items():
            if cat in value:
                sample_domain.append(key)
                train_data.loc[index, ("Domain")] = key
                break
    
    if len(sample_domain) < 1:
        print('----------------------------------------')
        print(row)


train_data.to_csv('/opt/data/private/dataset/AVVP/LLP_dataset/labels/AVVP_train_domain.csv', header=0, sep='\t')
test_data.to_csv('/opt/data/private/dataset/AVVP/LLP_dataset/labels/AVVP_test_domain.csv', header=0, sep='\t')
with open('/opt/data/private/dataset/AVVP/LLP_dataset/labels/domain_class.json', 'w') as f:
    json.dump(domain, f)
