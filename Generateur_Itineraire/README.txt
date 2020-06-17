- FindWay est le module qui contient tout le n�cessaire pour lire les fichiers, d�terminer les itin�raires et les afficher.
- Test_FindWay charge les fichiers RATP lorsqu'il est ex�cut� et la fonction Test() permet de visualiser les r�sultats
  de 12 itin�raires pr�sentants des particularit�s int�ressantes qui prouvent l'efficacit� de notre programme.
- Pour l'itin�raire Pont de S�vres - Saint-Fargeau, une carte folium est cr��e dans le r�pertoire du programme et s'appelle Itinerary.
- La lecture des fichiers prend environ 1 minute et ceux-ci doivent �tre dans le m�me r�pertoire que celui du programme.

Test_Station :
- Nous n'avons pas pu adapter notre code � Test_Station car la structure �tait trop diff�rente.
- La plupart des tests qui sont r�alis�s dans ce fichier le sont aussi dans notre fichier Test_FindWay.

Test_Network :
- Les itin�raires des quatre tests qui ne fonctionnent pas peuvent �tre affich�s avec la commande Test_Network() 
  qui se trouve dans le fichier Test_FindWay.
- Au d�but, la lecture des fichiers �taient impossible � cause des chemins des fichiers (slash dans le mauvais sens).
  Nous avons donc retir�s la partie "Inputs/" dans les noms des fichiers de Test_Network et cela a fonctionn�.

