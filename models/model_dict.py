from models.segment_anything.build_sam import sam_model_registry
from models.segment_anything_echosam.build_echosam import echosam_model_registry
from models.segment_anything_echosam_kp.build_echosam_kp import echosam_kp_model_registry

def get_model(modelname="SAM", args=None, opt=None):
    if modelname == "SAM":
        model = sam_model_registry['vit_b'](checkpoint=args.sam_ckpt)
    elif modelname == "EchoSAM":
        model = echosam_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    elif modelname == "EchoSAM_KP":
        model = echosam_kp_model_registry['vit_b'](args=args, checkpoint=args.sam_ckpt)
    else:
        raise RuntimeError("Could not find the model:", modelname)
    return model
