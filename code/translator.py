import numpy as np
import os
from googletrans import Translator
import csv

data_path = '../data'
bin_file = '../data/fr_emb.bin'

train_file = open(os.path.join(data_path, 'input_train.csv'),'r')

translator = Translator(service_urls=['translate.google.fr'])

input = '../data/input_train_trans.csv'
temporary = 'temp.csv'

#with open(input,'w') as temp :
#    pass
#with open(temporary ,'w') as temp :
#    pass

# with open(input ,'r') as begin :
#     with open(temporary,'w') as temp :
#         reader = csv.reader(train_file)
#         reader_begin = csv.reader(begin)
#         writer = csv.writer(temp)
#         lim=0
#         for begin_row in reader_begin :
#             writer.writerow(begin_row)
#             lim+=1
#             print('lim = ' + str(lim))
#         ind = 0
#         if lim == 0 :
#             writer.writerow(['ID','question'])
#         for row in reader:
#             if ind > lim :
#                 try:
#                     french = row[1]
#                     translation = translator.translate(french, 'en').text
#                     writer.writerow([str(ind),translation])
#                     print(ind)
#                 except:
#                     print('end at ', ind)
#                     break
#             ind +=1
#
# os.rename(temporary, input)

reader = csv.reader(train_file)
for row in reader:
    if row[0] == '431':
        french = row[1]
        print(french)


with open(input ,'r') as begin :
    with open(temporary,'w') as temp :
        reader_begin = csv.reader(begin)
        writer = csv.writer(temp)
        i=0
        writer.writerow(['ID','question'])
        for begin_row in reader_begin :
            if i > 0 :
                if i < 432 :
                    text = begin_row[1]
                    writer.writerow([str(i-1),text])
                elif i == 432 :
                    translation = translator.translate(french, 'en').text
                    writer.writerow([str(i-1),translation])
                    text = begin_row[1]
                    writer.writerow([str(i),text])
                    print(translation)
                else :
                    text = begin_row[1]
                    writer.writerow([str(i),text])
            i+=1
os.rename(temporary, input)
