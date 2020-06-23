Données :
Les fichiers "train.csv" et "train_results.csv" contiennent les données brutes.
Comme ils sont particulièrement long à charger par read_csv, le programme "Data_Preprocessing.R"
crée à partir des données brutes un nouveau fichier "training.csv" qui contient toutes les données
retravaillées et qui peut être chargé rapidemment dans Rstudio.

Exécution du R markdown :
Les résultats obtenus en exécutant le fichier markdown seront proches mais pas identiques
à ceux présentés dans le rapport. En effet, les échantillons train et test utilisés sont constitués
au hasard lors de l'exécution du programme.
L'encodage du markdown n'arrive pas à afficher les lettres avec accents.
Pour une version plus propre et complète du travail, se rapporter au rapport.

Nouvelle approche (fichier NBA_Challenge_New.py) :
Un an plus tard, dans le but d'améliorer les résultats, un autre modèle
de prédictions (LSTM) a été appliqué au jeu de données.