import clip
import torch
import torch.nn as nn
import torch.nn.functional as F


classes = [
    'apple',
    'aquarium fish',
    'baby',
    'bear',
    'beaver',
    'bed',
    'bee',
    'beetle',
    'bicycle',
    'bottle',
    'bowl',
    'boy',
    'bridge',
    'bus',
    'butterfly',
    'camel',
    'can',
    'castle',
    'caterpillar',
    'cattle',
    'chair',
    'chimpanzee',
    'clock',
    'cloud',
    'cockroach',
    'couch',
    'crab',
    'crocodile',
    'cup',
    'dinosaur',
    'dolphin',
    'elephant',
    'flatfish',
    'forest',
    'fox',
    'girl',
    'hamster',
    'house',
    'kangaroo',
    'keyboard',
    'lamp',
    'lawn mower',
    'leopard',
    'lion',
    'lizard',
    'lobster',
    'man',
    'maple tree',
    'motorcycle',
    'mountain',
    'mouse',
    'mushroom',
    'oak tree',
    'orange',
    'orchid',
    'otter',
    'palm tree',
    'pear',
    'pickup truck',
    'pine tree',
    'plain',
    'plate',
    'poppy',
    'porcupine',
    'possum',
    'rabbit',
    'raccoon',
    'ray',
    'road',
    'rocket',
    'rose',
    'sea',
    'seal',
    'shark',
    'shrew',
    'skunk',
    'skyscraper',
    'snail',
    'snake',
    'spider',
    'squirrel',
    'streetcar',
    'sunflower',
    'sweet pepper',
    'table',
    'tank',
    'telephone',
    'television',
    'tiger',
    'tractor',
    'train',
    'trout',
    'tulip',
    'turtle',
    'wardrobe',
    'whale',
    'willow tree',
    'wolf',
    'woman',
    'worm',
]
templates = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a black and white photo of a {}.',
    'a low contrast photo of a {}.',
    'a high contrast photo of a {}.',
    'a bad photo of a {}.',
    'a good photo of a {}.',
    'a photo of a small {}.',
    'a photo of a big {}.',
    'a photo of the {}.',
    'a blurry photo of the {}.',
    'a black and white photo of the {}.',
    'a low contrast photo of the {}.',
    'a high contrast photo of the {}.',
    'a bad photo of the {}.',
    'a good photo of the {}.',
    'a photo of the small {}.',
    'a photo of the big {}.',
]

class ResNet(nn.Module):
    def __init__(self, num_classes=100, layers=8,  type='RN50', device='cuda', T=0.1):
        super(ResNet, self).__init__()
        model, preprocess = clip.load(type, device)
        self.device = device
        self.T = T
        self.model = model.visual

        self.weight_energy = nn.Sequential(
            nn.Linear(1024, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

        imagenet_templates = templates


        class_name = classes
        self.text_embeddings = self.get_text_embeddings(class_name, imagenet_templates, model)

    def get_text_embeddings(self, classnames, templates, model):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [template.format(classname) for template in templates]  # format with class
                texts = clip.tokenize(texts).cuda()  # tokenize
                class_embeddings = model.encode_text(texts) # embed with text encoder
                class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
                class_embedding = class_embeddings.mean(dim=0)
                class_embedding = class_embedding / class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).float()
        return zeroshot_weights

    def uncertainty(self, value):
        energy_score = torch.logsumexp(value[:, :-1] / 1.0, 1)
        return self.MLP(energy_score.view(-1, 1))

    def logits_compute(self, image_embedding):
        text = self.text_embeddings.cuda()
        image_embedding = image_embedding / image_embedding.norm(dim=-1, keepdim=True)
        logits = image_embedding @ text
        return logits

    def visual_forward_one(self, x):
        def stem(x):
            x = self.model.relu1(self.model.bn1(self.model.conv1(x)))
            x = self.model.relu2(self.model.bn2(self.model.conv2(x)))
            x = self.model.relu3(self.model.bn3(self.model.conv3(x)))
            x = self.model.avgpool(x)
            return x

        x = x.type(self.model.conv1.weight.dtype)
        x = stem(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)

        return x

    def visual_forward_two(self, x):
        x = x.permute(1, 0, 2) # NLD -> LND
        for i in range(self.layers, 12):
            x = self.model.transformer.resblocks[i](x)

        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.model.ln_post(x[:, 0, :])

        if self.model.proj is not None:
            x = x @ self.model.proj
        return x

    def ft_blocks(self, x):
        x = self.model.layer4(x)
        x = self.model.attnpool(x)
        return x

    def forward(self, input, fc=False):
        if fc==True:
            output = self.logits_compute(input)
            score = self.weight_energy(input)
            return output, score

        input = self.visual_forward_one(input.half())
        feature = self.ft_blocks(input).float()
        output = self.logits_compute(feature)
        score = self.weight_energy(feature)
        return output, feature, score