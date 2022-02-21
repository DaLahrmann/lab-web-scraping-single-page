
from joblib import dump, load
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import pandas as pd
import numpy as np

secrets_file = open("../L.6.05/Spotify_Secret.txt","r")
string = secrets_file.read()
string.split('\n')

secrets_dict={}
for line in string.split('\n'):
    if len(line) > 0:
        secrets_dict[line.split(':')[0]]=line.split(':')[1]

sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=secrets_dict['cid'],
                                                           client_secret=secrets_dict['cs']))

flist=['danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','time_signature', 'duration_ms']
stitle=input('Input the Song Title')
artist=input('Input the Song Artists')

kmeans=load('KMeans_cluster.bin')
scalar=load('std_scaler.bin')
top100=pd.read_csv('top100.csv')
cluslist=pd.read_csv('hitlist_clustered.csv')

if (stitle in list(top100.title)) & (artist in list(top100[top100.title == stitle].interpret)):
    print(top100.sample(n=1))
else:
    results=sp.search(q=(stitle,artist))
    if results['tracks']['items'] !=[]:
        uri=results['tracks']['items'][0]['uri']
        features=sp.audio_features(uri)[0]
        if type(features)==dict:
            fs=[]
            for x in flist:
                fs.append(features[x])
            fs=np.array([fs,fs]) #scalar needs a 2D array to work
            fs=scalar.transform(fs)
            out=kmeans.predict(fs)
            out2=cluslist[cluslist.cluster == out[0]].sample(n=1)
            print(out2[['name','artist','uri']])
        else:
            print('Song not found')
    else: 
        print('Song not found')
        
    

