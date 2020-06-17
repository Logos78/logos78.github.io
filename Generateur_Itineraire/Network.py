import FindWay as fw

class Network(object) :

    def __init__(self, *file) :
        folder_list = []
        folder_list.extend(file)
        fw.Read_files(folder_list)

    def convert_date_time(self, dat) :
        return [str(dat.year*10000+dat.month*100+dat.day), "{}:{}:00".format(dat.hour,dat.minute)]


    def compute_shortest_path(self, station_i, station_f, date_and_time) :
        [date, hour] = self.convert_date_time(date_and_time)
        Itinerary = fw.Calcul_itinerary(station_i, station_f, date, hour)
        Parcours = [(Itinerary[0][0], 'transfer', 0,)]
        for i in Itinerary :
            if fw.display_dict[i[0]] == fw.display_dict[i[1]] :
                Parcours.append((i[1], 'transfer', int(i[2]/60),))
            else :
                Parcours.append((i[1], fw.display_line(i[1]), int(i[2]/60),))
        return [Parcours, int(Itinerary[-1][2]/60)]
