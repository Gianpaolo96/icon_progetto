# Python <stdio.h>
import warnings
import numpy as np
import pandas as pd
# Visualizzazione dati
import seaborn as sns
import matplotlib.pyplot as plt
# Pre-elaborazione
from sklearn.preprocessing import StandardScaler
# Predizione
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
# Clustering
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
# Cancella warnings
warnings.filterwarnings("ignore")
# Importa tutti i dati dal dataset
data = pd.read_csv('SpotifyFeatures.csv',encoding = "ISO-8859-1");
# Definire set di predizione
x = data[['Beats.Per.Minute', 'Energy', 'Danceability', 'Loudness.dB', 'Liveness', 'Valence', 'Length', 'Acousticness', 'Speechiness']]
# Definire variabile target
y = data[['Popularity']]
# Definire info variabili
info = data[['Track.Name', 'Artist.Name', 'Genre']]
# Correlazione Spearman
sc = pd.concat([x,y], axis=1).corr(method='spearman')
# Generare una maschera per il triangolo superiore
triangle_mask = np.zeros_like(sc, dtype=np.bool)
triangle_mask[np.triu_indices_from(triangle_mask)] = True
plt.figure(figsize = (25,10))
sns.heatmap(data = sc, linewidths=.1, linecolor='black', vmin = -1, vmax = 1, mask = triangle_mask, annot = True,
            cbar_kws={"ticks":[-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1]});
plt.yticks(rotation=45);
plt.savefig('1.png', bbox_inches='tight')
plt.figure(figsize = (25,10));
sns.countplot(x="Genre", data=info);
plt.ylabel('Songs Count');
plt.xticks(rotation=45, ha='right');
plt.savefig('2.png', bbox_inches='tight')
for i in range(0,len(info)):
    if info.loc[i,'Genre'] in ['electropop','dance pop','canadian pop','acoustic pop','australian pop',
                               'folk-pop','boy band','danish pop','pop']:
        info.loc[i,'Genre2'] = 'Pop'
    elif info.loc[i,'Genre'] in ['dfw rap','rap','chicago rap','cali rap','emo rap','melodic rap','gangster rap','pop rap']:
        info.loc[i,'Genre2'] = 'Rap'
    elif info.loc[i,'Genre'] in ['modern rock','classic rock','alternative rock']:
        info.loc[i,'Genre2'] = 'Rock'
    elif info.loc[i,'Genre'] in ['canadian hip hop','canadian contemporary r&b','neo soul','north carolina hip hop',
                                 'conscious hip hop','detroit hip hop','lgbtq+ hip hop']:
        info.loc[i,'Genre2'] = 'Hip hop'
    elif info.loc[i,'Genre'] in ['electro house','edm','big room','electropop','brostep']:
        info.loc[i,'Genre2'] = 'Eletro'
    elif info.loc[i,'Genre'] in ['latin']:
        info.loc[i,'Genre2'] = 'Latin'
plt.figure(figsize = (25,10));
sns.countplot(x="Genre2", data=info);
plt.ylabel('Songs Count');
plt.xlabel('New Genres');
plt.savefig('3.png', bbox_inches='tight')
df = pd.concat([info,y],axis=1)
plt.figure(figsize = (25,10));
sns.barplot(data=df,x='Genre2',y='Popularity');
plt.xticks(rotation=45, ha='right');
plt.savefig('4.png', bbox_inches='tight')
plt.figure(figsize = (25,10));
sns.barplot(data=df,x='Genre',y='Popularity');
plt.xticks(rotation=45, ha='right');
plt.savefig('5.png', bbox_inches='tight')
# Definire variabili
metricsRFR = []       # Prediction Scores for Random Forest Regressor
metricsSVR = []       # Prediction Scores for Support Vector Regressor
metricsKNR = []       # Prediction Scores for K Nearest Neighbours Regressor

# Inizializza pre-elaborazione
ss = StandardScaler()

# inizializza modelli
svr = SVR()
rfr = RandomForestRegressor()
knr = KNeighborsRegressor(n_neighbors=3)

# Cross-Validation 10 Fold
cv = KFold(n_splits=10, random_state=1206, shuffle=True)

# Loop into all 10 Folds
for train_index, test_index in cv.split(x):
    # Define train and test to simplify our life
    x_train, x_test, y_train, y_test = x.loc[train_index,:], x.loc[test_index,:], y.loc[train_index,:], y.loc[test_index,:]
    # Fit and Transform our train set using StandardScaler
    x_train = ss.fit_transform(x_train)
    # Transform our test set based on x_train
    x_test = ss.transform(x_test)
    # Fit Models
    rfr.fit(x_train, y_train)
    svr.fit(x_train, y_train)
    knr.fit(x_train, y_train)
    # Predict using our models
    y_pred_rfr = rfr.predict(x_test)
    y_pred_svr = svr.predict(x_test)
    y_pred_knr = knr.predict(x_test)
    # Calculate and append Prediction Mean Square Error
    metricsRFR.append(mean_squared_error(y_test.values.ravel(), y_pred_rfr))
    metricsSVR.append(mean_squared_error(y_test.values.ravel(), y_pred_svr))
    metricsKNR.append(mean_squared_error(y_test.values.ravel(), y_pred_knr))

# Print the results
print('RFR had Averaged MSE: %.2f\nSVR had Averaged MSE: %.2f\nKNR had Averaged MSE: %.2f'
      % (np.mean(metricsRFR), np.mean(metricsSVR), np.mean(metricsKNR)))
# Defining some variables
metricsRFR = []       # Prediction Scores for Random Forest Regressor
metricsSVR = []       # Prediction Scores for Support Vector Regressor
metricsKNR = []       # Prediction Scores for K Nearest Neighbours Regressor
rfrMSE = []           # Aux variable to keep MSE track for 3,4 and 5 features in Random Forest Regressor
svrMSE = []           # Aux variable to keep MSE track for 3,4 and 5 features in Support Vector Regressor
knrMSE = []           # Aux variable to keep MSE track for 3,4 and 5 features in K Nearest Neighbours Regressor
# Preprocessing init
ss = StandardScaler()

# Models init
svr = SVR()
rfr = RandomForestRegressor()
knr = KNeighborsRegressor(n_neighbors=3)

# Cross-Validation 10 Fold
cv = KFold(n_splits=10, random_state=1206, shuffle=True)

# Loop into all 10 Folds
for train_index, test_index in cv.split(x):
    # Define train and test to simplify our life
    x_train, x_test, y_train, y_test = x.loc[train_index,:], x.loc[test_index,:], y.loc[train_index,:], y.loc[test_index,:]
    # Fit and Transform our train set using StandardScaler
    x_train = ss.fit_transform(x_train)
    # Transform our test set based on x_train
    x_test = ss.transform(x_test)
    # Apply Feature Selection to use only 3,4 or 5 features
    for nFeat in [3,4,5]:
        # Init FS object to nFeat predictors variables
        fs = SelectKBest(score_func=mutual_info_regression, k=nFeat)
        # Choose our selected features in train/test set
        x_train_fs = fs.fit_transform(x_train, y_train.values.ravel())
        x_test_fs = fs.transform(x_test)
        # Fit Models
        rfr.fit(x_train_fs, y_train.values.ravel())
        svr.fit(x_train_fs, y_train.values.ravel())
        knr.fit(x_train_fs, y_train.values.ravel())
        # Predict using our models
        y_pred_rfr = rfr.predict(x_test_fs)
        y_pred_svr = svr.predict(x_test_fs)
        y_pred_knr = knr.predict(x_test_fs)
        # Save MSE
        rfrMSE.append(mean_squared_error(y_true=y_test.values.ravel(), y_pred=y_pred_rfr))
        svrMSE.append(mean_squared_error(y_true=y_test.values.ravel(), y_pred=y_pred_svr))
        knrMSE.append(mean_squared_error(y_true=y_test.values.ravel(), y_pred=y_pred_knr))

    # Append Prediction Mean Square Error
    metricsRFR.append(rfrMSE)
    metricsSVR.append(svrMSE)
    metricsKNR.append(knrMSE)
    # Reset our MSE lists
    rfrMSE = []
    svrMSE = []
    knrMSE = []
    # Calculate AVG MSE for each selected group feature in each model
    avgRFR = np.mean(metricsRFR, axis=0)
    avgSVR = np.mean(metricsSVR, axis=0)
    avgKNR = np.mean(metricsKNR, axis=0)
    # Print results
    print('--- AVG MSE for Random Forest Regressor')
    print('3 Features: %.2f\n4 Features: %.2f\n5 Features: %.2f\n' % (avgRFR[0], avgRFR[1], avgRFR[2]))
    print('--- AVG MSE for Support Vector Regressor')
    print('3 Features: %.2f\n4 Features: %.2f\n5 Features: %.2f\n' % (avgSVR[0], avgSVR[1], avgSVR[2]))
    print('--- AVG MSE for K Nearest Regressor')
    print('3 Features: %.2f\n4 Features: %.2f\n5 Features: %.2f\n' % (avgKNR[0], avgKNR[1], avgKNR[2]))
    # Scale our x data
    ss = StandardScaler()
    # Fit and transform our data using StandardScaler
    x_ss = ss.fit_transform(x)
    # Create a PC space with 2 components only
    pca = PCA(n_components=2)
    # Fit and Transform X to PC dimension
    pc = pca.fit_transform(x_ss)
    # Print Explained Variance Ratio
    print('PC1 explained %.2f ratio and PC2 explained %.2f ratio of total variance.' %
          (pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1]))
    # Define our range of clusters based on genre size
    n_clusters = range(2, len(info.Genre.unique()))
    # Create a list to append silhouette_avg values for each K to plot
    silhouette_avg = []

    # Loop to find optimal K
    for K in n_clusters:
        # Create KNN model
        km = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)
        # Fit and predict with K clusters using Principal Components
        pred = km.fit_predict(pc)
        # Calculate silhouette score
        silhouette_avg.append(silhouette_score(pc, pred))

    # Plot the results to define our optimal K
    plt.figure(figsize=(15, 10))
    plt.plot(n_clusters, silhouette_avg, marker='o');
    plt.xticks(n_clusters);
    plt.xlabel('Number of clusters');
    plt.ylabel('Silhouette Averaged Score');
    # Show the plot
    plt.savefig('6.png', bbox_inches='tight')
    # Create our KMeans model using optimal K
    km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)
    # Fit and predict with 2 clusters using Principal Components
    pred = km.fit_predict(pc)
    # Create a Dataframe for PC with target var
    df = pd.DataFrame(data=pc, columns=['PC1', 'PC2'])
    # Create a column for our clusters (KMeans prediction)
    df['Clusters'] = pd.Series(pred)
    # Join with target info/popularity dataset
    df = pd.concat([df, y, info], axis=1)
    # A sample of our new dataframe
    df.head(3)
    # Plot using seaborn our clusterization
    plt.figure(figsize=(20, 10));
    sns.scatterplot(x="PC1", y="PC2", hue="Clusters", data=df, palette=['green', 'blue'], s=50);
    # Show the plot
    plt.savefig('7.png', bbox_inches='tight')
    # Define subplots to get a side by side view
    fig, ax = plt.subplots(1, 2, figsize=(15, 5));
    # Plot KMeans
    p1 = sns.scatterplot(x="PC1", y="PC2", hue="Clusters", data=df, palette=['green', 'blue'], ax=ax[0], s=50);
    # Put KMeans legend outside
    p1.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
    # Plot with genre colors
    p2 = sns.scatterplot(x="PC1", y="PC2", hue="Genre2", data=df, ax=ax[1], s=50);
    # Show the plot
    plt.savefig('8.png', bbox_inches='tight')
    # Put again legend outside
    p2.legend(loc='center left', bbox_to_anchor=(1.25, 0.5), ncol=1);
    # Insert a safe space between the plots
    fig.tight_layout()
    # Show the plot
    plt.savefig('9.png', bbox_inches='tight')
    # Plot using seaborn our clusterization
    plt.figure(figsize=(20, 10));
    sns.scatterplot(x="PC1", y="PC2", hue="Clusters", data=df, palette=['green', 'blue'], size="Popularity",
                    sizes=(10, 300), edgecolor="black");
    # Show the plot
    plt.savefig('10.png', bbox_inches='tight')
    # Create a PC space with 3 components only
    pca = PCA(n_components=3)
    # Fit and Transform X to PC dimension
    pc = pca.fit_transform(x_ss)
    pca.explained_variance_ratio_
    # Print Explained Variance Ratio
    print('PC1 explained %.2f\nPC2 explained %.2f\nPC3 explained %.2f\nFor total of %.2f Explained Variance Ratio.' %
          (pca.explained_variance_ratio_[0], pca.explained_variance_ratio_[1], pca.explained_variance_ratio_[2],
           sum(pca.explained_variance_ratio_)))
    # Define our range of clusters based on genre size
    n_clusters = range(3, len(info.Genre.unique()))
    # Create a list to append silhouette_avg values for each K to plot
    silhouette_avg = []
    # Loop to find optimal K
    for K in n_clusters:
        # Create KNN model
        km = KMeans(n_clusters=K, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)
        # Fit and predict with K clusters using Principal Components
        pred = km.fit_predict(pc)
        # Calculate silhouette score
        silhouette_avg.append(silhouette_score(pc, pred))
    # Plot the results to define our optimal K
    plt.figure(figsize=(15, 10))
    plt.plot(n_clusters, silhouette_avg, marker='o');
    plt.xticks(n_clusters);
    plt.xlabel('Number of clusters');
    plt.ylabel('Silhouette Averaged Score');
    # Show the plot
    plt.savefig('11.png', bbox_inches='tight')
    # Create our KMeans model using optimal K
    km = KMeans(n_clusters=6, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=1206)
    # Fit and predict with 2 clusters using Principal Components
    pred = km.fit_predict(pc)
    # Create a Dataframe for PC with targets
    df = pd.DataFrame(data=pc, columns=['PC1', 'PC2', 'PC3'])
    # Create a column for our clusters (KMeans prediction)
    df['Clusters'] = pd.Series(pred)
    # Join with target info/popularity dataset
    df = pd.concat([df, y, info], axis=1)
    # A sample of our new dataframe
    df.head(3)
    # Plot 3D - KNN Clusters
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # x = np.array(df['PC1'])
    # y = np.array(df['PC2'])
    # z = np.array(df['PC3'])
    # ax.scatter(x, y, z, marker="s", c=df["Clusters"], s=40, cmap="RdBu")
    # plt.savefig("test.png")
    # Define ranges for TSNE hyperparameters
    perplexities = [5, 10, 15, 30, 40, 50, 60, 70, 80, 95]
    learning = [200, 800, 2000]
    # Create subplots
    fig, ax = plt.subplots(len(perplexities), len(learning), figsize=(20, 15))
    # Manual Grid Search for best hyper parameters through loop
    for p in range(0, len(perplexities)):
        for l in range(0, len(learning)):
            # Create TSNE model
            tsne = TSNE(n_components=2, n_iter=2000, perplexity=perplexities[p], learning_rate=learning[l])
            # Fit and Transform with X scaled for PC
            x_emb = tsne.fit_transform(x_ss)
            # Turn X embeddeb into a dataframe to a easy plot
            x_emb = pd.DataFrame(data=x_emb, columns=['XE1', 'XE2'])
            # Join with target categories
            x_emb = pd.concat([x_emb, info], axis=1)
            # Plot 2D data
            sns.scatterplot(x='XE1', y='XE2', hue="Genre2", data=x_emb, ax=ax[p][l]);
            ax[p][l].set_title(("Perplexity=%d | Learning=%d" % (perplexities[p], learning[l])));
            ax[p][l].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.);

    # Improve our layout
    plt.tight_layout()
# Show the plot
plt.savefig('12.png', bbox_inches='tight')