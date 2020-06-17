#!/usr/bin/env python
# coding: utf-8


#####################################
##### Creation of dictionnaries #####
#####################################


# Trips dictionnary
# Keys : trip_id
# Values : service_id

# Routes dictionnary
# Keys : trip_id
# Values : route_id
def read_trips(f,trips,routes) :
    trips_file = open(os.path.join(f,"trips.txt"),"r")
    trips_file.readline()
    for line in trips_file:
        a = line.split(",")
        trips[int(a[2])] = int(a[1])    # trip_id : service_id
        routes[int(a[2])] = int(a[0])   # trip_id : route_id
    trips_file.close()
    return [trips,routes]



# Line_direction dictionnary
# Keys : route_id
# Values : [line, direction]
def read_routes(f,d):
    routes_file = open(os.path.join(f,"routes.txt"),"r")
    routes_file.readline()
    for line in routes_file:
        a = line.split(",")
    # Clean and get the direction and line number
        line_number = a[2].strip('"')
        
        a[3] = a[3].split("<->")
        a[3][0]=a[3][0].strip('"').strip('(').strip() 
        a[3][1] = a[3][1].strip('"').replace(")", "")
        if a[3][1][-5:] == 'Aller':
            direction = a[3][1][:-8].strip()
        else:
            direction = a[3][0]

        d[int(a[0])] = [line_number, direction]
    
    routes_file.close()
    return d



# Service dictionnary
# Keys : dates
# Values : list of service_id asssociated to a date
def read_calendar_dates(f,d) :
    services_file = open(os.path.join(f,"calendar_dates.txt"),"r")
    services_file.readline()
    for line in services_file:
        a = line.split(",")
        if a[1] not in d :
            d[a[1]] = []
        d[a[1]].append(int(a[0]))
    services_file.close()
    return d



# Platform dictionnary (d)
# Keys : station's name
# Values : list of platform_id (i.e. stop_id) of a train station

# Display dictionnary (used to display station's names in the output)
# Keys : stop_id (platform 's number)
# Values :  station's name

# Coordinates dictionnary (used to display itinerary on Paris map)
# Keys : stop_id (platform 's number)
# Values :  GPS coordinates
def read_stops(f,d,display,coordinates) :
    platform_file = open(os.path.join(f,"stops.txt"),"r", encoding = "utf8")
    platform_file.readline()
    for line in platform_file :
        a = line.split(",")
        a[2] = a[2].strip('"')
        
    # Display dictionnary
        display[int(a[0])] = a[2]   # stop_id : station's name
        
    # Coordinates dictionnary
        if a[2]=="Montparnasse-Bienvenue" or a[2]=="Pelleport" or int(a[0])==1887:
            a[4] = a[5]
            a[5] = a[6]
        coordinates[int(a[0])] = [float(a[4]), float(a[5])]     # station's name : [latitude, longitude]
            
    # Platform dictionnary
        if a[2] not in d :
            d[a[2]] = []
        d[a[2]].append(int(a[0]))       # station's name : [platform_id]
      
    platform_file.close()
    return [d,display,coordinates]



# Convert time in format : "hh:mm:ss" into seconds
def convert_time(s) :
    return int(s[0])*3600 + int(s[1])*60 + int(s[2])

# Convert back time in seconds into the format "hh:mm:ss"
def convert_hour(i) :
    h = int(i/3600)
    m = int((i - 3600*h)/60)
    s = i - h*3600 - m*60
    if h >= 24 :
        h = h - 24
    if s == 0 :
        if m <= 10 :
            return "{}h 0{}m".format(h,m)
        else :
            return "{}h {}m".format(h,m)
    else :
        if m <= 10 :
            if s <= 10 :
                return "{}h 0{}m 0{}s".format(h,m,s)
            else :
                return "{}h 0{}m {}s".format(h,m,s)
        else :
            if s <= 10 :
                return "{}h {}m 0{}s".format(h,m,s)
            else :
                return "{}h {}m {}s".format(h,m,s)

# Better displaying for a date in format YYYYMMDD
def convert_date(date) :
    return "{}/{}/{}".format(date[6:], date[4:6], date[:4])



# A neighbour in the graph used in the Dijsktra algorithm
class neighbour(object) :
    def __init__(self, platform0, transfer0=0):
        self.platform_id = platform0    # stop_id of the platform     
        self.services = []              # List of service object
        self.transfer = transfer0       # Transfer time between two platforms of the same station
      

    def __repr__(self) :
        return str([self.platform_id, self.services, self.transfer])


# A train service
class service(object) :
    def __init__(self,service0) :
        self.service_id = service0      # service_id of the service
        self.departures = []            # List of departures to the neighbour from the previous station
                                        # Contain departure time and travel time to the neighbour

    # Add a schedule (with departure time and travel time) in the departure list
    # And sort it
    def add_path(self,departure0,arrival0) :
        self.departures.append([convert_time(departure0),convert_time(arrival0)-convert_time(departure0)])
        self.departures.sort()

    def __repr__(self) :
        return str([self.service_id,self.departures])


# Find or create (if it doesn't exist) a platform and return it
def find_platform(l, platform0) :
    # Look for platform0 in l = path_dict[i] and return it
    for i in l :
        if i.platform_id == platform0 :
            return i
    # If platform0 doesn't exist, it's created and returned
    l.append(neighbour(platform0))
    return l[-1]


# Find or create (if it doesn't exist) a platform and return it
def find_service(l,service0) :
    # Look for service0 in l = path_dict[i] and return it
    for i in l :
        if i.service_id == service0 :
            return i
    # If service0 doesn't exist, it's created and returned
    l.append(service(service0))
    return l[-1]



# Path dictionnary
# Keys : platform
# Values : list of platform's neighbours

# Stop_trip dictionnary (used to write the line and the direction)
# Keys : stop_id
# Values :  trip_id
def read_stop_times(f,path,stop_trip, trips_dict) :
    path_file = open(os.path.join(f,"stop_times.txt"),"r")
    progression = -1
    path_file.readline()
     
    for line in path_file:
        a = line.split(",")
        stop_trip[int(a[3])] = int(a[0])
        
        # If the platform isn't in path_dict, it's added with an empty list of neighbour
        if int(a[3]) not in path :     
            path[int(a[3])] = list()  
        
        if int(a[4])-progression == 1 :          # If it's the same trip
            # Add a schedule
            V = find_platform(path[prev_platform],int(a[3])) 
            s = find_service(V.services,trips_dict[int(a[0])])
            s.add_path(prev_departure,a[1].split(":"))

        # Save datas from this line
        prev_departure = a[2].split(":")        # departure_time
        prev_platform = int(a[3])               # stop_id 
        progression = int(a[4])                 # stop_sequence
        
    path_file.close()
    return [path,stop_trip]


# Read transfers.txt files to add transfer times between two platforms of the same station in path_dict
def read_transfers(f,d) :
    transfers_file = open(os.path.join(f,"transfers.txt"),"r")
    transfers_file.readline()
    for line in transfers_file:
        a = line.split(",")
        if (int(a[0]) in d) and (int(a[1]) in d) :
            d[int(a[0])].append(neighbour(int(a[1]),int(a[3])))
            d[int(a[1])].append(neighbour(int(a[0]),int(a[3])))
    transfers_file.close()
    return d



### Read all files
# The only folders in the current folder must be the metro lines ones 
import os
#folder_list = [i for i in os.listdir() if os.path.isdir(i)]

# Alternative way but require to know the names of the folders
lines_number = ["1", "2", "3", "3b", "4", "5", "6", "7", "7b", "8", "9", "10", "11", "12", "13", "14"]
files_name = "RATP_GTFS_METRO_"
folder_list = [files_name + i for i in lines_number]


def Read_files(folder_list) :
    # Creation of dictionnaries
    global trips_dict, routes_dict, service_dict, platform_dict, display_dict
    global coordinates_dict, path_dict, stop_trip, line_direction_dict
    trips_dict = {}
    routes_dict = {}
    service_dict = {}
    platform_dict = {}
    display_dict = {}
    coordinates_dict = {}
    path_dict = {}
    stop_trip = {}
    line_direction_dict = {}
    
    for f in folder_list :
        [trips_dict, routes_dict] = read_trips(f, trips_dict, routes_dict)
        service_dict = read_calendar_dates(f, service_dict)
        [platform_dict, display_dict, coordinates_dict] = read_stops(f, platform_dict, display_dict, coordinates_dict)
        [path_dict,stop_trip] = read_stop_times(f, path_dict, stop_trip, trips_dict)
        path_dict = read_transfers(f, path_dict)
        line_direction_dict = read_routes(f,line_direction_dict)

    # Exceptions (transfer is necessary but not defined in transfers.txt)
    if "RATP_GTFS_METRO_7" in folder_list :
        path_dict[2272].append(neighbour(1876,30))  # Maison Blanche (line 7)
    if "RATP_GTFS_METRO_7b" in folder_list :
        path_dict[2002].append(neighbour(2191,30))  # Botzaris (line 7B)
        path_dict[2434].append(neighbour(1756,30))  # Pré-Saint-Gervais (line 7B)
    if "RATP_GTFS_METRO_10" in folder_list :
        path_dict[2193].append(neighbour(2004,30))  # Boulogne Jean Jaurès (line 10)
        path_dict[1903].append(neighbour(2299,30))  # Javel André Citroën (line 10)
    if "RATP_GTFS_METRO_10" in folder_list :
        path_dict[2536].append(neighbour(1656,30))  # La Fourche (line 13)




##############################
##### Dijsktra algorithm #####
##############################

# Return train services scheduled at date0
def define_services(date0) :
    return service_dict[date0]

# Convert a hour0 string in format "hh:mm:ss" into seconds
def define_time(hour0) :
    return convert_time(hour0.split(":"))

# Return a stop_id of a platform in station0
def define_platform(station0) :
    return platform_dict[station0][0]

# Return the previous day
def define_previous_day(date) :
    if int(date[6:]) == 1 :
        if int(date[4:6]) in [2,4,6,8,9,11] :       # Back to a month with 31 days
            new_date = int(date) - 100 + 30
            new_date = str(new_date)

        elif int(date[4:6]) in [5,7,10,12] :        # Back to a month with 30 days
            new_date = int(date) - 100 + 29
            new_date = str(new_date)

        elif int(date[4:6]) == 3 :                  # March to February
            new_date = int(date) - 100 + 27
            new_date = str(new_date)

        else :                                      # January to December
            new_date = int(date) - 10000 + 1100 + 30
            new_date = str(new_date)
            
    else :                                          # Other days
        new_date = int(date) - 1
        new_date = str(new_date)

    return new_date


# Return the next day
def define_next_day(date) :
    if int(date[6:]) == 30 and (int(date[4:6]) in [4,6,9,11]) :     # From a month with 30 days
        new_date = int(date) + 100 - 29
        new_date = str(new_date)

    elif int(date[6:]) == 31 :                                      # From a month with 30 days
        new_date = int(date) + 100 - 30
        if int(date[4:6]) == 12 :                                   # December to January
             new_date = new_date - 1200
        new_date = str(new_date)

    elif int(date[6:]) == 28 and int(date[4:6]) == 2 :              # February to March
        new_date = int(date) + 100 - 27
        new_date = str(new_date)
        
    else :                                                          # Other days
        new_date = int(date) + 1
        new_date = str(new_date)

    return new_date


# Return platform's line
def display_line(stop_id):
    return line_direction_dict[routes_dict[stop_trip[stop_id]]][0]

# Return platform's direction
def display_direction(stop_id):
    return line_direction_dict[routes_dict[stop_trip[stop_id]]][1]



# Calcul the weight for a neighbour
def Calcul_weight(neighbour0,day_services,time,other_day_services) :
    # Create a second list of day services for the previous / next day
    day_services_bis = []
    if other_day_services != [] :
        s = other_day_services[0]                   # 1 for next day / -1 for previous day
        day_services_bis = other_day_services[1]    # The list of services
        
    if neighbour0.transfer == 0 :                   # The scoped platform and its neighbour are not in the same station
        Weights = []
        for j in neighbour0.services :              # For each train service,
            if j.service_id in day_services :       # check if the service is scheduled that day
                k = 0
                # Find the closest departure time from the current time (variable time) since departures is sorted
                while j.departures[k][0] - time < 0 and k < len(j.departures)-1:
                    k = k+1
                    
                if j.departures[k][0] - time >= 0 :  # If it's positive, then it's the next train
                    Weights.append([j.departures[k][0]-time, j.departures[k][1]])
                    
            elif j.service_id in day_services_bis :
                k = 0
                # Add or remove 86400 seconds (24 hours) to departure time
                while j.departures[k][0] + s*86400 - time < 0 and k < len(j.departures)-1:
                    k = k+1
                    
                if j.departures[k][0] + s*86400 - time >= 0 :
                    Weights.append([j.departures[k][0]+s*86400-time, j.departures[k][1]])
                    
        Weights.sort()
        
        if Weights == [] :         # If the platform has no neighbour or all the train already passed
            return 0
        else :
            return [sum(Weights[0]), Weights[0][0]+time]     # Return the minimal weight of all services

        
    else :                                       # The scoped platform and its neighbour are in the same station
        return [neighbour0.transfer, 0]          # Return walking time
    



def Calcul_itinerary(station_i,station_f,date,hour) :
### Define variables
    # Define time (in seconds)
    time = define_time(hour)

    # Define day services and add more from next/previous day if trip starts after 22h / before 02h
    if date not in service_dict.keys():
        print('PAS DE SERVICES')    # No services
        return False
    day_services = define_services(date)
    if time >= 79200 :
        other_day_services = [1, define_services(define_next_day(date))]
    elif time <= 7200 :
        other_day_services = [-1, define_services(define_previous_day(date))]
    else :
        other_day_services = []

    # Define the starting platform and add the opposité platform to its neighbour in path_dict
    current_platform = define_platform(station_i)
    platform_i_bool = False
    for p in platform_dict[station_i] :
        if p not in [i.platform_id for i in path_dict[current_platform]] and p != current_platform :
            path_dict[current_platform].append(neighbour(p,30))
            platform_i_bool = True
            platform_i = current_platform

    if current_platform in platform_dict[station_f] :
        print("VOUS Y ETES DEJA")   # You are already here
        return False
    
    # Variables used by Disjktra algorithm
    Dijsktra_tab = []
    Display_tab = []                        # Dijsktra_tab with the name of the station instead of stop_id
    current_time = time
    duration = 0                            # Total travel time at each step
    Already_visited = {current_platform}    # All nodes "visited" by the algorithm
    Path = []
    k = 1



### Execute Dijkstra algorithm 
    while current_platform not in platform_dict[station_f]:         # While we have not reached the final station,
        for p in path_dict[current_platform] :                      # check all neighbours of the current node
            if p.platform_id not in Already_visited :               # which have not being already visited
                c = Calcul_weight(p,day_services,current_time,other_day_services)     # and compute their weight
                if c != 0 :
                    # Append [weight, next_node, current_node, departure_time]
                    Dijsktra_tab.append([duration + c[0], p.platform_id, current_platform, c[1]])
                    #Display_tab.append([duration + c[0], display_dict[p.platform_id], p.platform_id, c[1]])
        Dijsktra_tab.sort()
        #Display_tab.sort()
        #print(Display_tab[:5]      # Display the five first nodes of Dijsktra_tab

        
        if Dijsktra_tab == [] :
            print("TRAJET IMPOSSIBLE")  # Trip is impossible
            return False

        else :
        # Update the informations about the new current node        
            current_platform = Dijsktra_tab[0][1]
            current_time = time + Dijsktra_tab[0][0]
            duration = Dijsktra_tab[0][0]
        # Add the path chosen to Path
            Path.append([Dijsktra_tab[0][2],current_platform, duration, Dijsktra_tab[0][3]])
        # Remove the current node from Dijsktra tab and marked it as already visited
            Already_visited.add(current_platform)
            Dijsktra_tab = Dijsktra_tab[1:]
            Display_tab = Display_tab[1:]
	
    if platform_i_bool == True :
        del path_dict[platform_i][-1]

    

### Reconstructs the correct itinerary from Path    
    min_duration = duration
    Itinerary = []
    
    # Start in current_platform, last element of Path (i.e. final platform)
    while current_platform not in platform_dict[station_i] :     
        for j in Path :
            if j[1] == current_platform and j[2] <= min_duration :       # Check all node connected to current_platform
                min_duration = j[2]                                      # Find the one with minimal total travel time (min_duration)
                next_platform = j[0]
                departure_time = j[3]   # Saved only for displaying
        Itinerary.append([next_platform, current_platform, min_duration, departure_time])
        current_platform = next_platform                                 # Repeat until the initial node has been reached
    Itinerary.reverse()

    return Itinerary





################################
### Display itinerary on map ###
################################

### Display itinerary with text
def Display_itinerary_text(Itinerary, hour) :
    if Itinerary == False :
        return None
    time = define_time(hour)
    
    print("Heure actuelle : ", hour, "\n")
    
    print("Ligne", display_line(Itinerary[0][0]))
    print("Direction :", display_direction(Itinerary[0][0]))
    print("Départ :", display_dict[Itinerary[0][0]], " à ", convert_hour(Itinerary[0][3]), '\n')

    change = False
    k = 0
    for i in Itinerary :
        if (display_dict[i[0]] == display_dict[i[1]]) :
            if change == True :     # Adjust the number of station in case of double change in same station
                k = k+1
            change = True
        elif change == True :
            print('\n')
            print("Changement - Ligne", display_line(i[0]))
            print("Direction :", display_direction(i[0]))
            print("Départ :", display_dict[i[0]], " à ", convert_hour(i[3]), '\n')
            change = False
            
        if change == False :
            print(display_dict[i[1]], "   ", convert_hour(time + i[2]))
            
    print('\n')
    print("Arrivée : ", display_dict[Itinerary[-1][1]], " à ", convert_hour(time + Itinerary[-1][2]))
    print("Nombre de stations : ",len(Itinerary)-k)
    print("Durée : ", convert_hour(Itinerary[-1][2] + time - Itinerary[0][3]))



### Display itinerary on map
import folium

def Display_itinerary(Itinerary) :
    if Itinerary == False :
        return None
    
    # Create map of Paris and add departure and arrival points
    Map = folium.Map(location=[48.852347,2.3181518], zoom_start=12)
    folium.Marker(coordinates_dict[Itinerary[0][0]], tooltip = "Départ : {} à {}".format(display_dict[Itinerary[0][0]], convert_hour(Itinerary[0][3])), \
                    icon=folium.Icon(icon='home')).add_to(Map)
    folium.Marker(coordinates_dict[Itinerary[-1][1]], tooltip = "Arrivée : {} à {}".format(display_dict[Itinerary[-1][1]], convert_hour(Itinerary[-1][3])), \
                    icon = folium.Icon(icon='flag')).add_to(Map)

    # Display itinerary
    coordinates_list=[]
    colors_list=[]
    color = 1;
    for t in Itinerary:
        coordinates_list.append(coordinates_dict[t[0]])
        if display_dict[t[0]] == display_dict[t[1]]:    # Change of line
            color += 1
            #connecting point
            folium.Marker(coordinates_dict[t[1]], \
                        tooltip = "{} / Changement - Ligne {}, Direction {}".format(display_dict[t[1]], display_line(t[1]), display_direction(t[1]), convert_hour(t[3])), \
                        icon = folium.Icon(icon = 'transfer')).add_to(Map)       
        colors_list.append(color)

    coordinates_list.append(coordinates_dict[t[1]])
    folium.ColorLine(coordinates_list, colors=colors_list, weight = 10, opacity = 1).add_to(Map)
    Map.save('Itinerary.html')




