import logging

from fedml_api.model.cv.cnn import CNN_DropOut
from fedml_api.model.cv.cnn_custom import CNNSmall, CNNMedium, CNNLarge, CNNCustomLayers
from fedml_api.model.cv.generator import ImageGenerator, ConditionalImageGenerator
from fedml_api.model.cv.mobilenet import mobilenet
from fedml_api.model.cv.resnet import resnet56
from fedml_api.model.cv.resnet_gn import resnet18
from fedml_api.model.linear.lr import LogisticRegression
from fedml_api.model.nlp.rnn import RNN_OriginalFedAvg, RNN_StackOverFlow

cv_model_builders = {
    'cnn_small': CNNSmall,
    'cnn_medium': CNNMedium,
    'cnn_large': CNNLarge,
    'cnn_custom': CNNCustomLayers
}

cv_datasets = {'mnist': 1, 'femnist': 1, 'fed_cifar100': 3, 'cinic10': 3, 'cifar10': 3, 'cifar100': 3, 'emnist': 1}
cv_datasets_image_size = {'mnist': [1, 32, 32], 'emnist': [1, 32, 32], 'femnist': [1, 28, 28],
                          'fed_cifar100': [3, 24, 24], 'cinic10': [3, 32, 32], 'cifar10': [3, 32, 32],
                          'cifar100': [3, 32, 32]}


def create_model(args, model_name, output_dim, layers=None):
    logging.info("create_model. model_name = %s, output_dim = %s" % (model_name, output_dim))
    model = None
    if model_name == "lr" and args.dataset == "mnist":
        logging.info("LogisticRegression + MNIST")
        model = LogisticRegression(28 * 28, output_dim)
    elif model_name == "cnn" and args.dataset == "femnist":
        logging.info("CNN + FederatedEMNIST")
        model = CNN_DropOut(False)
    elif model_name == "resnet18_gn" and args.dataset == "fed_cifar100":
        logging.info("ResNet18_GN + Federated_CIFAR100")
        model = resnet18()
    elif model_name == "rnn" and args.dataset == "shakespeare":
        logging.info("RNN + shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "rnn" and args.dataset == "fed_shakespeare":
        logging.info("RNN + fed_shakespeare")
        model = RNN_OriginalFedAvg()
    elif model_name == "lr" and args.dataset == "stackoverflow_lr":
        logging.info("lr + stackoverflow_lr")
        model = LogisticRegression(10000, output_dim)
    elif model_name == "rnn" and args.dataset == "stackoverflow_nwp":
        logging.info("RNN + stackoverflow_nwp")
        model = RNN_StackOverFlow()
    elif model_name == "resnet56":
        model = resnet56(class_num=output_dim)
    elif model_name == "mobilenet":
        model = mobilenet(class_num=output_dim)
    elif model_name == 'generator':
        model = ImageGenerator(args.nz, args.ngf, nc=cv_datasets[args.dataset], img_size=32)
    elif model_name == 'conditional_generator':
        model = ConditionalImageGenerator(num_classes=output_dim, nz=args.nz, ngf=args.ngf,
                                          nc=cv_datasets[args.dataset], img_size=32)
    elif args.dataset in cv_datasets:
        model = cv_model_builders[model_name](cv_datasets[args.dataset], output_dim,
                                              cv_datasets_image_size[args.dataset], layers)
    return model


def create_local_models_from_config(client_model_config, args, dataset):
    client_models = []
    client_num = 0

    for entry in client_model_config['client_models']:
        model_name = entry['model']
        layers = None
        if model_name == 'cnn_custom':
            assert 'layers' in entry.keys(), 'Missing required field "layers": [int]'
            layers = entry['layers']

        model = create_model(args, model_name=entry['model'], output_dim=dataset[7], layers=layers)
        client_models.append((model, entry['freq']))
        client_num += entry['freq']

    if args.dataset in ['cifar10', 'cifar100', 'mnist', 'emnist', 'cinic10']:
        assert args.client_num_in_total == client_num
    logging.info(client_models)

    return client_models
