
class Config:

    def __init__(self):
        #data configure
        self.batch_size = 1
        self.shuffle = True
        self.loadThread = 4
        self.inputsize = (121,145,121) # (30,30,30)
        #training configure
        self.epochs = 11
        self.num_classes = 1
        self.seed = 0
        self.data_mode = "single" # "single" or "multi"
        #multiple testing configure
        self.threshold = 0.1
        self.replications = 1  # matches the sample_size for x
        self.cluster_number = 0
        self.sample_number = 0



