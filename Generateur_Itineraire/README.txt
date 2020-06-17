- FindWay est le module qui contient tout le nécessaire pour lire les fichiers, déterminer les itinéraires et les afficher.
- Test_FindWay charge les fichiers RATP lorsqu'il est exécuté et la fonction Test() permet de visualiser les résultats
  de 12 itinéraires présentants des particularités intéressantes qui prouvent l'efficacité de notre programme.
- Pour l'itinéraire Pont de Sèvres - Saint-Fargeau, une carte folium est créée dans le répertoire du programme et s'appelle Itinerary.
- La lecture des fichiers prend environ 1 minute et ceux-ci doivent être dans le même répertoire que celui du programme.

Test_Station :
- Nous n'avons pas pu adapter notre code à Test_Station car la structure était trop différente.
- La plupart des tests qui sont réalisés dans ce fichier le sont aussi dans notre fichier Test_FindWay.

Test_Network :
- Les itinéraires des quatre tests qui ne fonctionnent pas peuvent être affichés avec la commande Test_Network() 
  qui se trouve dans le fichier Test_FindWay.
- Au début, la lecture des fichiers étaient impossible à cause des chemins des fichiers (slash dans le mauvais sens).
  Nous avons donc retirés la partie "Inputs/" dans les noms des fichiers de Test_Network et cela a fonctionné.

