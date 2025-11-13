import clip
from torchvision.transforms import transforms


def get_pretrained_model(args):
    if args.model == 'clip_vitbase16':
        model, preprocessing = clip.load("ViT-B/16", device=args.device)

    elif args.model == 'clip_vitbase32':
        model, preprocessing = clip.load("ViT-B/32", device=args.device)

    elif args.model == 'clip_vitlarge14':
        model, preprocessing = clip.load("ViT-L/14", device=args.device)

    elif args.model == 'clip_rn50':
        model, preprocessing = clip.load("RN50", device=args.device)

    elif args.model == 'clip_rn101':
        model, preprocessing = clip.load("RN101", device=args.device)

    else:
        raise ValueError(f"Unknown model {args.model}. ")

    return model, preprocessing
