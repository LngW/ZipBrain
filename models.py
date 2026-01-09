import torch

def load_model_by_args(args, records):

    records['model'] = [
        ('name', args.model),
        ('in_channels', args.in_channels),
        ('n_classes', args.n_classes),
    ]

    if args.model in ['SPaRCNet', 'ContraWR', 'CNNTransformer', 'FFCL', 'STTransformer', 'BIOT']:
        from thirdparty.BIOT.model import (
            SPaRCNet,
            ContraWR,
            CNNTransformer,
            FFCL,
            STTransformer,
            BIOTClassifier,
        )

        if args.model == "SPaRCNet":
            model = SPaRCNet(
                in_channels=args.in_channels,
                sample_length=int(args.sampling_rate * args.sample_length),
                n_classes=args.n_classes,
                block_layers=4,
                growth_rate=16,
                bn_size=16,
                drop_rate=0.5,
                conv_bias=True,
                batch_norm=True,
            )

        elif args.model == "ContraWR":
            model = ContraWR(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.token_size,
                steps=args.hop_length // 5,
            )

        elif args.model == "CNNTransformer":
            model = CNNTransformer(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.sampling_rate,
                steps=args.hop_length // 5,
                dropout=0.2,
                nhead=4,
                emb_size=256,
            )

        elif args.model == "FFCL":
            model = FFCL(
                in_channels=args.in_channels,
                n_classes=args.n_classes,
                fft=args.token_size,
                steps=args.hop_length // 5,
                sample_length=int(args.sampling_rate * args.sample_length),
                shrink_steps=20,
            )

        elif args.model == "STTransformer":
            model = STTransformer(
                emb_size=256,
                depth=4,
                n_classes=args.n_classes,
                channel_legnth=int(
                    args.sampling_rate * args.sample_length
                ),  # (sampling_rate * duration)
                n_channels=args.in_channels,
            )

        elif args.model == "BIOT":
            model = BIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
            records['model'] += [
                ('token_size', args.token_size),
                ('hop_length', args.hop_length),
            ]

            if args.pretrain_model_path and (args.sampling_rate == 200):
                model.biot.load_state_dict(torch.load(args.pretrain_model_path))
                print(f"load pretrain model from {args.pretrain_model_path}")
    elif args.model in ['CBIOT', 'TKBIOT', 'SABIOT', 'ToMeBIOT']:

        from model import (
            TopKBiotClassifier, 
            ToMeBiotClassifier, 
            CusBIOTClassifier
        )

        if args.model == "CBIOT":
            model = CusBIOTClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
            )
            records['model'] += [
                ('token_size', args.token_size),
                ('hop_length', args.hop_length),
            ]

        elif args.model == "TKBIOT":
            model = TopKBiotClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
                k=args.k
            )
            records['model'] += [
                ('token_size', args.token_size),
                ('hop_length', args.hop_length),
                ('k', args.k)
            ]

        elif args.model == "SABIOT":
            model = TopKBiotClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
                k = 0
            )
            records['model'] += [
                ('token_size', args.token_size),
                ('hop_length', args.hop_length),
            ]

        elif args.model == "ToMeBIOT":
            model = ToMeBiotClassifier(
                n_classes=args.n_classes,
                # set the n_channels according to the pretrained model if necessary
                n_channels=args.in_channels,
                n_fft=args.token_size,
                hop_length=args.hop_length,
                r = args.k
            )
            records['model'] += [
                ('token_size', args.token_size),
                ('hop_length', args.hop_length),
                ('r', args.k)
            ]
    else:
        raise NotImplementedError
    
    return model
