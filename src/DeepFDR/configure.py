
class Config:
    #data configure
    batch_size = 1
    shuffle = True
    loadThread = 4
    inputsize = (30,30,30) #(121,145,121) # (30,30,30)
    #training configure
    epochs = 17
    num_classes = 1
    seed = 0
    #multiple testing configure
    threshold = 0.1
    replications = 5 # matches the sample_size for x
    cluster_number = 0
    sample_number = 0



