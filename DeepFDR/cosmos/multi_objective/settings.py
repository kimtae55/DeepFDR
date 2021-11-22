import random
# datsets override methods override generic

#
# datasets
#
multi_mnist = dict(
    dataset='multi_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=2,
    alpha=1.2,
)

deepFDR = dict(
    dataset='deepFDR',
    dim=(1, 32, 32, 32),
    objectives=['MSELoss', 'MSELoss'],
    lamda=2,        # 
    alpha=1.2,      #
)

multi_fashion = dict(
    dataset='multi_fashion',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=2,
    alpha=1.2,
)

multi_fashion_mnist = dict(
    dataset='multi_fashion_mnist',
    dim=(1, 36, 36),
    objectives=['CrossEntropyLoss', 'CrossEntropyLoss'],
    lamda=8,
    alpha=1.2,
)

#
# methods
#
cosmos = dict(
    method='cosmos',
    lamda=2,        # Default for multi-mnist
    alpha=1.2,      #
)

SingleTaskSolver = dict(
    method='SingleTask',
    num_starts=2,   # two times for two objectives (sequentially)
)

uniform_scaling = dict(
    method='uniform',
)

#
# Common settings
#
generic = dict(    
    # Seed.
    seed=1,
    
    # Directory for logging the results
    logdir='result',

    # dataloader worker threads
    num_workers=4,

    # Number of test preference vectors for Pareto front generating methods    
    n_test_rays=25,

    # Evaluation period for val and test sets (0 for no evaluation)
    eval_every=5,

    # Evaluation period for train set (0 for no evaluation)
    train_eval_every=0,

    # Checkpoint period (0 for no checkpoints)
    checkpoint_every=10,

    # Use a multi-step learning rate scheduler with defined gamma and milestones
    use_scheduler=True,
    scheduler_gamma=0.1,
    scheduler_milestones=[20,40,80,90],

    # Number of train rays for methods that follow a training preference (ParetoMTL and MGDA)
    num_starts=1,

    # Training parameters
    lr=1e-3,
    batch_size=32,
    epochs=100,

    # Reference point for hyper-volume calculation
    reference_point=[2, 2],
)
