import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.decomposition import PCA
from gensim.models import word2vec

model = word2vec.Word2Vec.load('./wiki.model')

words = []
words.append(["生物","r"])
words.append(["数学","r"])
words.append(["物理","r"])
words.append(["豚肉","c"])
words.append(["鶏肉","c"])
words.append(["キャベツ","c"])
words.append(["チーズ","c"])
words.append(["牛乳","c"])
words.append(["卵","c"])
words.append(["白菜","c"])
words.append(["じゃがいも","c"])
words.append(["ニンジン","c"])
words.append(["外務省","y"])
words.append(["バス停","m"])
words.append(["電車","m"])
words.append(["新幹線","m"])
words.append(["バス","m"])
words.append(["タクシー","m"])
words.append(["車","m"])
words.append(["自転車","m"])

length = len(words)
data = []
j = 0
while j < length:
    data.append(model[words[j][0]])
    j += 1

pca = PCA(n_components=2)
pca.fit(data)
data_pca= pca.transform(data)

length_data = len(data_pca)

font_path = '/usr/share/fonts/truetype/takao-gothic/TakaoGothic.ttf'
fp = FontProperties(fname=font_path)

i = 0
while i < length_data:
    #点プロット
    plt.plot(data_pca[i][0], data_pca[i][1], ms=5.0, zorder=2, marker="x", color=words[i][1])
 
    #文字プロット
    plt.annotate(words[i][0], (data_pca[i][0], data_pca[i][1]), size=7, fontproperties=fp)
 
    i += 1
 
plt.show()
