import numpy as np

rng = np.random.default_rng()

base_config = dict(
    gen_type="AdainVC",
    # gen_type="LORD",
    content_dim=512,
    class_dim=256,
    # samples=np.array([rng.integers(0,high=1114),rng.integers(1114,high=2170),rng.integers(2170,high=3000),rng.integers(3000,high=4000)]),
    # samples=np.array([1100, 1644, 2635, 3505]),
    samples=np.array([620,1784,2355,3355]),
    content_std=1,
    content_decay=6e-6,

    n_adain_layers=2,
    adain_dim=32,


    # decoder params
    c_in=128,
    c_cond=256,
    c_h=256,
    c_out=128,
    kernel_size=7,
    n_conv_blocks=13,
    upsample=[2, 1, 1, 4,1,1, 2, 1, 1, 2, 1, 1, 1],
    act="relu",
    sn=False,
    dropout_rate=0.0,
    l1_weight=0.1,
    perceptual_loss=dict(
        layers=[2, 3, 5, 8, 11],
        layers_vggish=[4, 5, 7, 8, 11, 13]

    ),
    loss_types=[
        "image",
        # "sparsity",
        "Rho",
        # "Vggish",
        # "constractive"
        "sp_loss",
    ],
    loss_weights=dict(
        image_loss=0.3,
        sound_loss=1.1,
        sp_loss=0.08,
        sparse_loss=0.02,
        adv_loss=0.5
    ),

    train=dict(
        batch_size=64,
        n_epochs=130,

        learning_rate=dict(
            generator=1e-3,
            latent=1e-2,
            min=2e-7,
            disc=1e-5
        )
    ),

    train_encoders=dict(
        batch_size=64,
        n_epochs=120,
        
        
        instrument_encoder=dict(),
        content_encoder=dict(),
        
        learning_rate=dict(
            max=1e-3,
            min=5e-6
        )
    ),
)
