import clip


def get_pretrained_model(args, verified_checkpoint_path=None):
    verified_checkpoint_path = verified_checkpoint_path or getattr(
        args, 'verified_checkpoint_path', None
    )
    if verified_checkpoint_path is not None:
        # A verified run must load the exact file that was hashed. Passing a
        # symbolic name here would let package metadata redirect the loader.
        return clip.load(str(verified_checkpoint_path), device=args.device)

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
