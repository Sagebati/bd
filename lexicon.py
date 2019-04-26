import pandas as pd

word2senti=pd.read_csv('./lexicon/SentiWordNet_3.0.0.txt',sep='\t', skiprows=25, engine='python')

word2senti['words']=word2senti['SynsetTerms'].apply(lambda x : re.split('#\d ?',x)[:-1])

{word2senti2.update({word2senti['words'][x][i]:{'pos':max(word2senti['PosScore'][x],word2senti2.get(word2senti['words'][x][i],{'pos':0})['pos']), 'neg':max(word2senti['NegScore'][x],word2senti2.get(word2senti['words'][x][i], {'neg':0})['neg']), 'id':word2senti['ID'][x]}}) for x in word2senti.drop(['# POS','SynsetTerms','Gloss'],axis=1).T.to_dict() if len(word2senti['words'][x]) >=1 for i in range(len(word2senti['words'][x]))}

#s contient l'ensemble des mots distincts
s=[]
for i in corpus: #corpus est à définir
    s=list(set(s+i.split()))

w2z={}
w2s={x: word2senti2.get(x, {'pos':100, 'neg':100}) for x in s}

#nb de mot qui ne sont pas dans le dictionnaire
len([x for x in w2s if w2s[x]['pos']==100 and w2s[x]['neg']==100])

#nb de mot avec un sentiments
len([x for x in w2s if not(w2s[x]['pos']==0 and w2s[x]['neg']==0)])
