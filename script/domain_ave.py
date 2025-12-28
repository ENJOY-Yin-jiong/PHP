import pandas as pd
import os
import json


domain = dict()

domain["human activities"] = ["Male speech, man speaking", "Female speech, woman speaking", "Frying (food)", "Baby cry, infant cry", "Toilet flush"]
domain["animal activities"] = ["Bark", "Goat", "Cat", "Horse", "Rodents, rats, mice"]
domain["music performances"] = ["Violin, fiddle", "Flute", 'Ukulele', 'Shofar', 'Acoustic guitar', 'Banjo', 'Accordion', 'Mandolin']
domain["vehicle sounds"] = ["Fixed-wing aircraft, airplane", "Race car, auto racing", 'Helicopter', 'Truck', 'Motorcycle', 'Train horn', 'Bus']
domain["domestic environments"] = ["Church bell", "Clock", "Chainsaw"]


df = pd.read_csv(os.path.join( "/opt/data/private/dataset/AVE/labels/Annotations.txt"), sep="&")


for index, row in df.iterrows():
    sample_domain = None
    category = row["Category"]
    
    for key, value in domain.items():
        if category in value:
            sample_domain = key
            # row["Domain"] = key
            df.loc[index, ("Domain")] = key
            break
    
    if sample_domain is None:
        print("------------------------------------------------")
        print(row["Category"])

df.to_csv('/opt/data/private/dataset/AVE/labels/Annotations_with_domain.txt',sep='&',index=0)
with open('domain_class.json', 'w') as f:
    json.dump(domain, f)