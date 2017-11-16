import torch

opts = {
    # Configurations
    'gpu': True and torch.cuda.is_available(),

    # Paths
    'vgg_model_path': '../model/imagenet-vgg-m.mat',
    'model_sl_path': '../model/actnet-sl.pth',
    'model_rl_path': '../model/actnet-rl.pth',
    'lstm_path': '../model/lstm.pth',

    # Supervised learning
    'batch_frames': 1,
    'batch_pos': 1,
    'batch_neg': 1,
    'batch_size': 8,
    'overlap_pos': [0.7, 1],
    'overlap_neg': [0, 0.5],

    # Gradient descent
    'lr': 1e-4,
    'w_decay': 5e-4,
    'momentum': 0.9,
    'grad_clip': 10,

    # ActNet
    'trainable_layers': ['conv', 'fc'],
    'num_past_actions': 10,
    'num_actions': 11,
    'img_size': 107,
    'padding': 16,

    # Tracking
    'epsilon': 0.1,
    'epsilon_decay': 0.1,
    'max_actions': 20,
    'alpha': 0.03,
    'iou_criterion': 0.7,
    'reward_discount': 0.99,

    # GAE
    'gamma': 0.99,
    'tau': 0.99,
    'entropy_coeff': 0.99,
    'value_loss_coeff': 0.99
}

kwargs = {
    'num_workers': 1,
    'pin_memory': True
} if opts['gpu'] else {}
