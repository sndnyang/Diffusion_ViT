from .vit import ViT
from .cait import CaiT
from .pit import PiT
from .swin import SwinTransformer
from .t2t import T2T_ViT
from .DifViT import ViT as DifViT
from .CVT import ViT as CVT


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(
                current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


def create_model(img_size, n_classes, args):
    if args.model == 'vit':
        patch_size = 4 if img_size == 32 else 8
        heads = 12 if args.dim % 12 == 0 else 8
        model = ViT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=args.dim,
                    mlp_dim_ratio=2, depth=9, heads=heads, dim_head=args.dim // heads,
                    stochastic_depth=args.sd, is_SPT=args.SPT, is_LSA=args.LSA)

    elif args.model == 'cait':       
        patch_size = 4 if img_size == 32 else 8
        model = CaiT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, stochastic_depth=args.sd,
                     is_LSA=args.LSA, is_SPT=args.SPT)
        
    elif args.model == 'pit':
        patch_size = 2 if img_size == 32 else 4    
        args.channel = 96
        args.heads = (2, 4, 8)
        args.depth = (2, 6, 4)
        dim_head = args.channel // args.heads[0]
        
        model = PiT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=args.channel,
                    mlp_dim_ratio=2, depth=args.depth, heads=args.heads, dim_head=dim_head, 
                    stochastic_depth=args.sd, is_SPT=args.SPT, is_LSA=args.LSA)

    elif args.model == 't2t':
        model = T2T_ViT(img_size=img_size, num_classes=n_classes, drop_path_rate=args.sd, is_SPT=args.SPT, is_LSA=args.LSA)
        
    elif args.model == 'swin':
        depths = [2, 6, 4]
        num_heads = [3, 6, 12]
        mlp_ratio = 2
        window_size = 4
        patch_size = 2 if img_size == 32 else 4
            
        model = SwinTransformer(img_size=img_size, window_size=window_size, drop_path_rate=args.sd, 
                                patch_size=patch_size, mlp_ratio=mlp_ratio, depths=depths, num_heads=num_heads, num_classes=n_classes, 
                                is_SPT=args.SPT, is_LSA=args.LSA)

    elif args.model == 'difvit':
        patch_size = 4 if img_size == 32 else 8
        if 'ps' in vars(args):
            patch_size = args.ps
        heads = 12 if args.dim % 12 == 0 else 8
        if args.heads is not None:
            heads = args.heads
            wlog(f'heads {heads}')
        depth = 9

        if args.depth is not None:
            depth = args.depth
            wlog(f'depth {depth}')

        model = DifViT(img_size=img_size, patch_size=patch_size, num_classes=n_classes, dim=args.dim,
                       mlp_dim_ratio=2, depth=depth, heads=heads, dim_head=args.dim // heads, channels=3 if args.dataset != 'mnist' else 1,
                       stochastic_depth=0, is_SPT=args.SPT, is_LSA=args.LSA, ffn_time=args.ffnt)


    else:
        assert False
    return model
