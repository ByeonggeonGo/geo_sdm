print('okok')

class Rich():
    def __init__ (self, rich_did, lu_rch_did, ru_rch_did, stream_flux, c_area, geometry):
        self.rich_did = rich_did
        self.lu_rch_did = lu_rch_did
        self.ru_rch_did = ru_rch_did
        self.stream_flux = stream_flux               ## 단위 m3/s
        self.c_area = c_area
        self.geometry = geometry
        self.stream_flux_estimated = None
        self.lu_child = None
        self.ru_child = None
        self.stp_family = None
        self.stp_site = False

        self.parent = None

        self.velocity = None                         ## 단위 m/s
        self.velocity_estimated = None
        self.rch_len = None
        self.r_time = None                           ## 단위 hour

        self.PEC_est = None
        self.mass_g = None


    
    def take_lu_child(self,lu_rich_class,):
        self.lu_child = lu_rich_class
        

    def take_ru_child(self,ru_rich_class,):
        self.ru_child = ru_rich_class

    def estimate_flux(self, parents_true_flux, parents_true_carea):
        # ~np.isnan(self.stream_flux_estimated)
        if self.stream_flux is None:
            self.stream_flux_estimated = parents_true_flux *(self.c_area/parents_true_carea)

        elif self.stream_flux is not None:
            print('참값 있음')