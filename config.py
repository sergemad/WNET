class Config():
    def __init__(self) -> None:
        self.debug = True
        self.input_size = 224 # Side length of square image patch
        self.batch_size = 10
        self.val_batch_size = 4
        self.test_batch_size = 1
        self.verbose_testing = True

        self.k = 30 #64 Number of classes
        self.num_epochs = 10 #250 for real
        self.data_dir = "./VOC2007/JPEGImages"
        self.showdata = False # Debug the data augmentation by showing the data we're training on.

        self.useInstanceNorm = False # Instance Normalization
        self.useBatchNorm = True # Only use one of either instance or batch norm
        self.useDropout = True
        self.drop = 0.65

        # Each item in the following list specifies a module.
        # Each item is the number of input channels to the module.
        # The number of output channels is 2x in the encoder, x/2 in the decoder.
        self.encoderLayerSizes = [64, 128, 256, 512]
        self.decoderLayerSizes = [1024, 512, 256]

        self.showSegmentationProgress = True
        self.segmentationProgressDir = './progress/'


        self.saveModel = True