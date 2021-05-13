import splitfolders

splitfolders.ratio('E:/data',
            output="D:/output2", seed=1337, ratio=(.8, 0.1, 0.1),
            group_prefix=None)
