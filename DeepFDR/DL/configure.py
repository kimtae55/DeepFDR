
class Config:

    def __init__(self):
        #data configure
        self.batch_size = 32
        self.shuffle = True
        self.loadThread = 4
        self.inputsize = [30,30,30]
        self.datapath = "../data/wnet/direct_x/sigma_125_1/"

        #training configure
        self.epochs = 1
        self.num_classes = 1
        self.seed = 0

