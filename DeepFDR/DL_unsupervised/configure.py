
class Config:

    def __init__(self):
        #data configure
        self.batch_size = 32
        self.shuffle = True
        self.loadThread = 4
        self.inputsize = (30,30,30)
        self.datapath = "/scratch/tk2737/DeepFDR/data/sim/test_dl/data_sim_direct_x.npz"
        self.labelpath = "/scratch/tk2737/DeepFDR/data/sim/label.npy"
        #training configure
        self.epochs = 30
        self.num_classes = 1
        self.seed = 0
        self.data_mode = "multi" # "single" or "multi"


