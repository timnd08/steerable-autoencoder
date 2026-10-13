import torch.nn as nn
import torch.nn.parallel as parallel

from . import vgg, vgg_original

def BuildAutoEncoder(args):

    if args.arch in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        configs = vgg.get_configs(args.arch)
        model = vgg.VGGAutoEncoder(configs)

    elif args.arch in ["vgg_original"]:
        configs = vgg_original.get_configs("vgg16")
        model = vgg_original.VGGAutoEncoder(configs)
    
    else:
        return None
    
    if args.parallel == 1:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = parallel.DistributedDataParallel(
                        model.to(args.gpu),
                        device_ids=[args.gpu],
                        output_device=args.gpu,
                        find_unused_parameters=True
                    )   
    
    else:
        model = nn.DataParallel(model).cuda()

    return model