opts = {
    # Configurations
    'gpu': True,

    # Paths
    'vgg_model_path': '../model/imagenet-vgg-m.mat',
    'model_path': '../model/actnet-vot-otb.pth',

    # Supervised learning
    'batch_frames': 1,
    'batch_pos': 1,
    'batch_neg': 1,
    'overlap_pos': [0.7, 1],
    'overlap_neg': [0, 0.5],

    # Gradient descent
    'lr': 1e-4,
    'w_decay': 5e-4,
    'momentum': 0.9,
    'grad_clip': 10,
    'n_cycles': 50,

    # ActNet
    'trainable_layers': ['conv', 'fc'],
    'img_size': 107,
    'padding': 16,

    # Tracking
    'epsilon': 0.1,
    'epsilon_decay': 0.1,
    'max_actions': 20,
    'alpha': 0.03
}
