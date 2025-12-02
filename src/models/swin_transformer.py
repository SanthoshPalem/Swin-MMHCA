import timm

def get_swin_transformer(model_name='swin_tiny_patch4_window7_224', pretrained=True, in_chans=3, **kwargs):
    """
    Returns a Swin Transformer model from the timm library.
    """
    model = timm.create_model(model_name, pretrained=pretrained, in_chans=in_chans, **kwargs)
    return model

if __name__ == '__main__':
    # Example of how to use the function
    swin = get_swin_transformer(features_only=True)
    print(swin)