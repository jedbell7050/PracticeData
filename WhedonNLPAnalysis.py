from NLPAnalysis import NLPAnalysis
import matplotlib.pyplot as plt 
# nltk.download('punkt')
#nltk.download('stopwords')

SoullessScript = open('ASoullessScript', 'r').read()
EntropyScript = open('BEntropyScript','r').read()

Soulless = NLPAnalysis(SoullessScript)
Entropy = NLPAnalysis(EntropyScript)


print(Soulless.bow(20))
print(Entropy.bow(20))
S = Soulless.word_len()
E = Entropy.word_len()
s = Soulless.sent_len()
e = Entropy.sent_len()
    

fig, ax = plt.subplots(2,2)

ax[0,0].hist(S, bins=30)
ax[0,1].hist(E, bins=30)
ax[0,0].set_title('Soulless')
ax[0,0].set_xlabel('Word Lengths')
ax[0,0].set_ylabel('Frequency')
ax[0,1].set_title('Entropy')
ax[0,1].set_xlabel('Word Lengths')


ax[1,0].hist(s, bins=30)
ax[1,1].hist(e, bins=30)
ax[1,0].set_xlabel('Sentence Lengths')
ax[1,0].set_ylabel('Frequency')
ax[1,1].set_xlabel('Sentence Lengths')

plt.show()