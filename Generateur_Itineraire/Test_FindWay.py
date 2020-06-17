import FindWay as fw
from FindWay import Calcul_itinerary
from FindWay import Display_itinerary_text
from FindWay import Display_itinerary


### Create dictionnaries
lines_number = ["1", "2", "3", "3b", "4", "5", "6", "7", "7b", "8", "9", "10", "11", "12", "13", "14"]
files_name = "RATP_GTFS_METRO_"
folder_list = [files_name + i for i in lines_number]

fw.Read_files(folder_list)
print("Write Test() in console to run 12 basic tests")
print("Write Test_Network() to search for itineraries wrote in Test_Network\n")



#############
### Tests ###
#############

def Test() :
    # One line
    print("######## TEST 1 ########")
    print("ONE LINE")
    date = '20190421'
    hour = '12:20:00'
    station_i = "Jussieu"
    station_f = "Pyramides"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")


    # Multiple changes
    print("######## TEST 2 ########")
    print("MULTIPLE CHANGES")
    date = '20190521'
    hour = '18:45:00'
    station_i = "Ecole Militaire"
    station_f = "Arts-et-Métiers"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    Display_itinerary(Itinerary)
    print("########################\n\n\n")


    # Starting in a terminus
    print("######## TEST 3 ########")
    print("STARTING IN A TERMINUS")
    date = '20190314'
    hour = '09:32:00'
    station_i = "Pont de Sèvres"
    station_f = "Saint-Fargeau"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    #Display_itinerary(Itinerary)
    print("########################\n\n\n")


    # End of day and month
    print("######## TEST 4 ########")
    print("END OF DAY AND MONTH")
    date = '20190228'
    hour = '23:55:00'
    station_i = "Cadet"
    station_f = "Couronnes"
    print("Date : ", fw.convert_date(date))
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")


    # After midnight
    print("######## TEST 5 ########")
    print("AFTER MIDNIGHT")
    print("Schedules are written in 24:XX:XX format for trains that left \nfrom their itinial station before midnight but the algorithm still works")
    date = '20190601'
    hour = '00:10:00'
    station_i = "Lourmel"
    station_f = "Bastille"
    print("Date : ", fw.convert_date(date))
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")


    # Early in the morning
    print("######## TEST 6 ########")
    print("EARLY IN THE MORNING")
    date = '20190124'
    hour = '03:20:00'
    station_i = "Ternes"
    station_f = "Varenne"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")

    # Fork
    print("######## TEST 7 ########")
    print("FORK")
    date = '20190715'
    hour = '13:30:00'
    station_i = "Mairie d'Ivry"
    station_f = "Villejuif-Louis Aragon"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")

    # Faster to change line
    print("######## TEST 8 ########")
    print("FASTER TO CHANGE LINES")
    print("Porte de la Villette and Place d'Italie both belong to line 7\nbut it's faster to change line")
    date = '20190403'
    hour = '14:33:00'
    station_i = "Place d'Italie"
    station_f = "Porte de la Villette"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")

    # No services
    print("######## TEST 9 ########")
    print("NO SERVICES FOR THIS DAY")
    date = '20190915'
    hour = '13:30:00'
    station_i = "Jussieu"
    station_f = "Censier-Daubenton"
    print("date : ", fw.convert_date(date))
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")

    # Impossible travel
    print("######## TEST 10 ########")
    print("NO MORE TRAIN THIS DAY")
    date = '20190228'
    hour = '23:55:00'
    station_i = "Blanche"
    station_f = "Bolivar"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")

    # Already here
    print("######## TEST 11 ########")
    print("ALREADY HERE")
    date = '20190506'
    hour = '13:30:00'
    station_i = "Jussieu"
    station_f = "Jussieu"
    print("Starting station : ", station_i)
    print("Final_station : ", station_f)
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")

    # Walking time underestimated
    print("######## TEST 12 ########")
    print("WALKING TIME UNDERESTIMATED")
    print("Shorter than RATP result because walking time not taken into account\nor underestimasted")
    date = '20190314'
    hour = '09:30:00'
    station_i = "Porte de Bagnolet"
    station_f = "Jussieu"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")





#############################
### Tests of Test_Network ###
#############################
def Test_Network() :
    # test_One_route_5_compute_shortest_path_with_weight
    print("Test_One_route_5_compute_shortest_path_with_weight")
    date = '20190324'
    hour = '23:25:00'
    station_i = "Place d\'Italie"
    station_f = "République"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n\n")


    # test_One_route_4_compute_shortest_path_with_weight
    print("test_One_route_4_compute_shortest_path_with_weight")
    date = '20190324'
    hour = '22:25:00'
    station_i = "Réaumur-Sébastopol"
    station_f = "Montparnasse-Bienvenue"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n")


    # test_Two_routes_4_5_compute_shortest_path_with_weight
    print("test_Two_routes_4_5_compute_shortest_path_with_weight")
    date = '20190321'
    hour = '15:25:00'
    station_i = "Château d'Eau"
    station_f = "Jacques-Bonsergent"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n")


    # test_All_Subway_routes_compute_shortest_path_with_weight
    print("test_All_Subway_routes_compute_shortest_path_with_weight")
    date = '20190321'
    hour = '15:25:00'
    station_i = "Château d'Eau"
    station_f = "Jacques-Bonsergent"
    Itinerary = Calcul_itinerary(station_i,station_f,date,hour)
    Display_itinerary_text(Itinerary, hour)
    print("########################\n\n")
