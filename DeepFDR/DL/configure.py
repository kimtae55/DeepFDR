
class Config:

    def __init__(self):
        #data configure
        self.batch_size = 32
        self.shuffle = True
        self.loadThread = 4
        self.inputsize = [30,30,30]
        self.datapath = "../data/wnet/"

        #training configure
        self.epochs = 100
        self.num_classes = 1
        self.seed = 0

        # early stop conditions
        self.patience = 10
        self.threshold = 1e-3

        #results configure
        self.savepath = self.datapath + "wnet.pth"
        self.savepath_loss = self.datapath + "loss.png"

        # test
        self.test_input = self.datapath + "test/5000.npy"
        self.test_output = ""
        self.gamma_orig = self.datapath + "test_label/5000.npy"
        self.label_datapath = self.datapath + "label/label.txt"
