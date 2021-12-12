import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
from model.adain_vc_model import SpeakerEncoder, ContentEncoder, Decoder
from Vggish.vggish import get_vggish_model
import torchaudio
import matplotlib.pyplot as plt
from cola import network, constants
import tensorflow as tf
from cpjku_dcase19.models.cp_resnet import get_model_based_on_rho
from config import base_config as config
import wandb


def plot_sp(sp):
    plt.imshow(sp, cmap='inferno')
    plt.colorbar()
    plt.show()


class LatentModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        self.content_embedding = RegularizedEmbedding(config['n_imgs'], config['content_dim'], config['content_std'])
        self.class_embedding = nn.Embedding(config['n_classes'], config['class_dim'])
        if config["gen_type"]=="LORD":
            self.modulation = Modulation(config['class_dim'], config['n_adain_layers'], config['adain_dim'])
            self.decoder=Generator( config['content_dim'], config["n_adain_layers"], config["adain_dim"], (128,128,1))
        elif  config["gen_type"]=="AdainVC":
            self.decoder = Decoder(c_in=config["c_in"], c_cond=config["c_cond"], c_h=config["c_h"], c_out=config["c_out"],
                               kernel_size=config["kernel_size"],
                               n_conv_blocks=config["n_conv_blocks"], upsample=config["upsample"], act=config["act"], sn=False,
                               dropout_rate=config["dropout_rate"])
        else:
            raise ModuleNotFoundError("decoder not implemented")

    def forward(self, img_id, class_id):
        content_code = self.content_embedding(img_id)
        class_code = self.class_embedding(class_id)
        if config["gen_type"]=="LORD":
            class_adain_params = self.modulation(class_code)
            generated_img = self.decoder(content_code, class_adain_params)
        elif  config["gen_type"]=="AdainVC":
            # matching dims from LORD to AdaIN-VC decoder
            content_code = content_code.reshape((-1, 128, (self.config["content_dim"]//128)))
            generated_img = self.decoder(content_code, class_code)
        return {
            'img': generated_img,
            'content_code': content_code,
            'class_code': class_code
        }

    def init(self):
        self.apply(self.weights_init)

    @staticmethod
    def weights_init(m):
        if isinstance(m, nn.Embedding):
            nn.init.uniform_(m.weight, a=-0.05, b=0.05)


class AmortizedModel(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.config = config
        # self.content_encoder=Encoder(config['img_shape'], config['content_dim'])
        # self.class_encoder=Encoder(config['img_shape'], config['class_dim'])
        self.content_encoder = ContentEncoder(c_in=128,
                                              c_h=128,
                                              c_out=128,
                                              kernel_size=5,
                                              bank_size=10,
                                              bank_scale=1,
                                              c_bank=128,
                                              n_conv_blocks=7,
                                              subsample=[4, 2, 1, 1,4, 1, 1],
                                              act="lrelu",
                                              dropout_rate=0.0)
        self.class_encoder = SpeakerEncoder(c_in=128,
                                            c_h=128,
                                            c_out=256,
                                            kernel_size=5,
                                            bank_size=8,
                                            bank_scale=1,
                                            c_bank=128,
                                            n_conv_blocks=6,
                                            n_dense_blocks=7,
                                            subsample=[1, 2, 1, 2, 1, 1],
                                            act="lrelu",
                                            dropout_rate=0.0)
        self.decoder =  Decoder(c_in=config["c_in"], c_cond=config["c_cond"], c_h=config["c_h"], c_out=config["c_out"],
                               kernel_size=config["kernel_size"],
                               n_conv_blocks=config["n_conv_blocks"], upsample=config["upsample"], act=config["act"], sn=False,
                               dropout_rate=config["dropout_rate"])

    def forward(self, img):
        return self.convert(img, img)

    def convert(self, content_img, class_img):
        # print("input ",content_img.shape)
        content_code = self.content_encoder(content_img)
        class_code = self.class_encoder(class_img)
        content_code = content_code.reshape((-1, 128, 4))

        # print("content_code ",content_code.shape)
        # print("class_code ",class_code.shape)
        generated_img = self.decoder(content_code, class_code)

        return {
            'img': generated_img,
            'content_code': content_code,
            'class_code': class_code
        }


class RegularizedEmbedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, stddev):
        super().__init__()

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.stddev = stddev

    def forward(self, x):
        x = self.embedding(x)

        if self.training and self.stddev != 0:
            noise = torch.zeros_like(x)
            noise.normal_(mean=0, std=self.stddev)

            x = x + noise

        return x


class Modulation(nn.Module):

    def __init__(self, code_dim, n_adain_layers, adain_dim):
        super().__init__()

        self.__n_adain_layers = n_adain_layers
        self.__adain_dim = adain_dim

        self.adain_per_layer = nn.ModuleList([
            nn.Linear(in_features=code_dim, out_features=adain_dim * 2)
            for _ in range(n_adain_layers)
        ])

    def forward(self, x):
        adain_all = torch.cat([f(x) for f in self.adain_per_layer], dim=-1)
        adain_params = adain_all.reshape(-1, self.__n_adain_layers, self.__adain_dim, 2)

        return adain_params


class Generator(nn.Module):

    def __init__(self, content_dim, n_adain_layers, adain_dim, img_shape):
        super().__init__()

        self.__initial_height = img_shape[0] // (2 ** n_adain_layers)
        self.__initial_width = img_shape[1] // (2 ** n_adain_layers)
        self.__adain_dim = adain_dim

        self.fc_layers = nn.Sequential(
            nn.Linear(
                in_features=content_dim,
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 8)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 8),
                out_features=self.__initial_height * self.__initial_width * (adain_dim // 4)
            ),

            nn.LeakyReLU(),

            nn.Linear(
                in_features=self.__initial_height * self.__initial_width * (adain_dim // 4),
                out_features=self.__initial_height * self.__initial_width * adain_dim
            ),

            nn.LeakyReLU()
        )

        self.adain_conv_layers = nn.ModuleList()
        for i in range(n_adain_layers):
            self.adain_conv_layers += [
                nn.Upsample(scale_factor=(2, 2)),
                nn.Conv2d(in_channels=adain_dim, out_channels=adain_dim, padding=1, kernel_size=3),
                nn.LeakyReLU(),
                AdaptiveInstanceNorm2d(adain_layer_idx=i)
            ]

        self.adain_conv_layers = nn.Sequential(*self.adain_conv_layers)

        self.last_conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=adain_dim, out_channels=64, padding=2, kernel_size=5),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=img_shape[2], padding=3, kernel_size=7),
            nn.Sigmoid()
        )

    def assign_adain_params(self, adain_params):
        for m in self.adain_conv_layers.modules():
            if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
                m.bias = adain_params[:, m.adain_layer_idx, :, 0]
                m.weight = adain_params[:, m.adain_layer_idx, :, 1]

    def forward(self, content_code, class_adain_params):
        self.assign_adain_params(class_adain_params)

        x = self.fc_layers(content_code)
        x = x.reshape(-1, self.__adain_dim, self.__initial_height, self.__initial_width)
        x = self.adain_conv_layers(x)
        x = self.last_conv_layers(x)

        return x


class Encoder(nn.Module):

    def __init__(self, img_shape, code_dim):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU()
        )

        self.fc_layers = nn.Sequential(
            nn.Linear(in_features=4096, out_features=256),
            nn.LeakyReLU(),

            nn.Linear(in_features=256, out_features=256),
            nn.LeakyReLU(),

            nn.Linear(256, code_dim)
        )

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv_layers(x)
        x = x.view((batch_size, -1))
        x = self.fc_layers(x)
        return x


class AdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, adain_layer_idx):
        super().__init__()
        self.weight = None
        self.bias = None
        self.adain_layer_idx = adain_layer_idx

    def forward(self, x):
        b, c = x.shape[0], x.shape[1]

        x_reshaped = x.contiguous().view(1, b * c, *x.shape[2:])
        weight = self.weight.contiguous().view(-1)
        bias = self.bias.contiguous().view(-1)

        out = F.batch_norm(
            x_reshaped, running_mean=None, running_var=None,
            weight=weight, bias=bias, training=True
        )

        out = out.view(b, c, *x.shape[2:])
        return out


class SpectralConvergenceLoss(torch.nn.Module):

    def __init__(self):
        super(SpectralConvergenceLoss, self).__init__()

    def forward(self, x_mag, y_mag):
        return torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")


def get_rho_model():
    model=get_model_based_on_rho(rho=5, config_only=False)
    cpt = torch.load(
        "/cs/labs/dina/matanhalfon/appliedDL/cpjku_dcase19/pretrained_models/single_resnet/last_model_345.pth")
    model.load_state_dict(cpt["state_dict"])
    return model


class Rho_model_features(nn.Module):
    def __init__(self, layer_ids):
        super().__init__()
        self.vggnet =get_rho_model()
        self.vggnet.requires_grad_(False)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        x = self.vggnet.in_c(x)
        output.append(x)
        x=self.vggnet.stage1(x)
        output.append(x)
        x = self.vggnet.stage2(x)
        output.append(x)
        x = self.vggnet.stage3(x)
        output.append(x)
        return output



class RhoDistance(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vgg = Rho_model_features(layer_ids)
        self.layer_ids = layer_ids

    def forward(self, s1, s2):
        b_sz = s1.size(0)
        s1=(s1-torch.mean(s1))/torch.std(s1)
        s2=(s2-torch.mean(s2))/torch.std(s2)
        s1 = torch.cat((s1, s1), dim=1)
        s2 = torch.cat((s2, s2), dim=1)
        f1 = self.vgg(s1)
        f2 = self.vgg(s2)
        loss = torch.abs(s1 - s2).view(b_sz, -1).mean(1)
        # loss=0
        for i in range(len(f1)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        return loss.mean()


class NetVGGFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = models.vgg16(pretrained=True)
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGDistance(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vgg = NetVGGFeatures(layer_ids)
        self.vgg.requires_grad_(False)
        self.layer_ids = layer_ids

    def forward(self, I1, I2):
        # To apply VGG on grayscale, we duplicate the single channel
        if I1.ndim == 3:
            I1 = torch.stack((I1, I1, I1), dim=1)
        elif I1.shape[1] == 1:
            I1 = torch.cat((I1, I1, I1), dim=1)

        if I2.ndim == 3:
            I2 = torch.stack((I2, I2, I2), dim=1)
        elif I2.shape[1] == 1:
            I2 = torch.cat((I2, I2, I2), dim=1)

        b_sz = I1.size(0)
        f1 = self.vgg(I1)
        f2 = self.vgg(I2)

        loss = torch.abs(I1 - I2).view(b_sz, -1).mean(1)
        # loss = 0
        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        return loss.mean()


def get_ssl_encoder():
    ckpt_path = "/cs/labs/dina/matanhalfon/appliedDL/cola_model/gtzan/cola_pretrain_test"
    ssl_network = network.get_contrastive_network(
        embedding_dim=512,
        temperature=0.2,
        similarity_type=constants.SimilarityMeasure.DOT,
        pooling_type="max")
    ssl_network.load_weights(
        tf.train.latest_checkpoint(ckpt_path)).expect_partial()
    ssl_network.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    encoder = ssl_network.embedding_model.get_layer("encoder")
    inputs = tf.keras.layers.Input(
        shape=(None, 64, 1))
    x = encoder(inputs)
    return encoder


class constractiveDistance(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = get_ssl_encoder()

    # self.mel_bank=torchaudio.transforms.MelScale(n_mels=64)
    # self.to_db=torchaudio.transforms.AmplitudeToDB()

    def forward(self, s1, s2):
        b_sz = s1.size(0)
        # s1_mel=self.mel_bank(torchaudio.functional.DB_to_amplitude(s1,1,1))
        # s1_mel=self.to_db(s1_mel)
        # s2_mel=self.mel_bank(torchaudio.functional.DB_to_amplitude(s2,1,1))
        # s2_mel=self.to_db(s2_mel)
        # print
        # swaped=torch.swapaxes(s2,1,3)[:,:,:64,...].cpu().detach().numpy()
        # print(swaped.shape)
        s1_mel = tf.convert_to_tensor((torch.swapaxes(s1, 1, 3)[:, :, :64, ...] / 10).cpu().detach().numpy(),
                                      dtype=tf.float32)
        s2_mel = tf.convert_to_tensor((torch.swapaxes(s2, 1, 3)[:, :, :64, ...] / 10).cpu().detach().numpy(),
                                      dtype=tf.float32)
        f1 = torch.Tensor(self.encoder(s1_mel).numpy())
        f2 = torch.Tensor(self.encoder(s2_mel).numpy())
        loss=torch.nn.functional.mse_loss(f1,f2)
        # loss = torch.square(f1 - f2).view(b_sz, -1).mean(1)
        return loss.mean()


class NetVGGishFeatures(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vggnet = get_vggish_model()
        self.layer_ids = layer_ids

    def forward(self, x):
        output = []
        for i in range(self.layer_ids[-1] + 1):
            x = self.vggnet.features[i](x)

            if i in self.layer_ids:
                output.append(x)

        return output


class VGGishDistance(nn.Module):

    def __init__(self, layer_ids):
        super().__init__()

        self.vgg = NetVGGishFeatures(layer_ids=layer_ids)
        self.layer_ids = layer_ids

    # self.mel_bank=torchaudio.transforms.MelScale(n_mels=64)
    # self.to_db=torchaudio.transforms.AmplitudeToDB()

    def forward(self, s1, s2, log=False, epoch=0):
        b_sz = s1.size(0)
        # s1_mel=self.mel_bank(torchaudio.functional.DB_to_amplitude(s1,1,1))
        # s1_mel=self.to_db(s1_mel)
        # s2_mel=self.mel_bank(torchaudio.functional.DB_to_amplitude(s2,1,1))
        # s2_mel=self.to_db(s2_mel)
        # s1_mel=torch.flip(s1,[2])
        # s2_mel=torch.flip(s2,[2])
        if log:
            wandb.log({f'log_mel-{epoch}': [wandb.Image(s1)]}, step=epoch)
            wandb.log({f'log_mel-{epoch}': [wandb.Image(s2)]}, step=epoch)
        f1 = self.vgg(s1)
        f2 = self.vgg(s2)
        loss = torch.abs(s1 - s2).view(b_sz, -1).mean(1)
        # loss=0
        for i in range(len(self.layer_ids)):
            layer_loss = torch.abs(f1[i] - f2[i]).view(b_sz, -1).mean(1)
            loss = loss + layer_loss

        return loss.mean()


class reconstraction_criteria(nn.Module):

    def __init__(self,config: dict):
        super().__init__()
        self.config=config
        self.loss_type=self.config["loss_types"]
        self.spectrul_loss = None if  "sp_loss" not in self.loss_type else SpectralConvergenceLoss()
        self.image_loss=None if "image" not in self.loss_type else VGGDistance(self.config['perceptual_loss']['layers'])
        self.sound_loss =None if ("Rho" not in self.loss_type and "Vggish" not in self.loss_type)\
            else( RhoDistance([]) if "Rho" in self.loss_type else VGGishDistance(self.config['perceptual_loss']['layers_vggish']) )
        self.ssl_loss= None if "constractive" not in self.loss_type else constractiveDistance()
        #flag to enforce sparsity
        self.sparsity_loss=False if "sparsity" not in self.loss_type else True

    def forward(self, out, batch):
        loss=0
        loss_loger=dict()
        if self.image_loss:
            image_loss=self.image_loss(out, batch)
            loss_loger["image_loss"]=image_loss
            loss+=image_loss*self.config['loss_weights']["image_loss"]
        if self.sound_loss:
            sound_loss=self.sound_loss(out[:,None,...], batch[:, None, ...])
            loss_loger["sound_loss"]=sound_loss
            loss+=sound_loss*self.config['loss_weights']["sound_loss"]
        if self.spectrul_loss:
            spectral_loss = self.spectrul_loss(out, batch)
            loss_loger["sp_loss"]=spectral_loss
            loss+=spectral_loss*self.config['loss_weights']["sp_loss"]
        if self.sparsity_loss:
            fro = torch.norm(out, p="fro")
            loss_loger["sparsity_loss"]=fro
            loss+=fro*self.config['loss_weights']["sparse_loss"]
        if self.ssl_loss:
            constractive_loss=self.ssl_loss(out[:, None, ...], batch[:, None, ...])
            loss_loger["ssl_loss"]=constractive_loss
            loss+=constractive_loss*self.config['loss_weights']["ssl"]
        loss_loger["rec_loss"]=loss
        return loss, loss_loger



class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int
    ):
        super(ConvBlock, self).__init__()

        self.__conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(2e-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out=self.__conv(x)
        return out

class Discriminator(nn.Module):
    def __init__(
            self,
            in_channels: int
    ):
        super(Discriminator, self).__init__()

        conv_channels = [
            (in_channels, 32),
            (32, 64),
            (64, 96),
            (96, 128),
        ]

        stride = 2

        nb_layer = 4

        self.__conv = nn.Sequential(*[
            ConvBlock(
                conv_channels[i][0],
                conv_channels[i][1]
            )
            for i in range(nb_layer)
        ])

        nb_time = 128
        nb_freq = 128
        n_classes=5
        out_size = (conv_channels[-1][1] * nb_time // stride ** nb_layer * nb_freq // stride ** nb_layer)
        self.__clf = nn.Linear(out_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_conv = self.__conv(x)
        out = out_conv.flatten(1, -1)
        # meraged=torch.cat([out,classes],1)
        out_clf = self.__clf(out)
        return out_clf

    def gradient_penalty(self, x_real: torch.Tensor, x_gen: torch.Tensor) -> torch.Tensor:
        device = "cuda" if next(self.parameters()).is_cuda else "cpu"

        batch_size = x_real.size()[0]
        eps = torch.rand(batch_size, 1, 1, 1, device=device)

        x_interpolated = eps * x_real + (1 - eps) * x_gen

        out_interpolated = self(x_interpolated)

        gradients = torch.autograd.grad(
            out_interpolated, x_interpolated,
            grad_outputs=torch.ones(out_interpolated.size(), device=device),
            create_graph=True, retain_graph=True
        )

        gradients = gradients[0].view(batch_size, -1)
        gradients_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradients_norm - 1.) ** 4.).mean()

        grad_pen_factor = 12.

        return grad_pen_factor * gradient_penalty


def discriminator_loss(y_real: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.log2(y_real) + torch.log2(1. - y_fake))


def generator_loss(y_fake: torch.Tensor) -> torch.Tensor:
    return -torch.mean(torch.log2(y_fake))


def wasserstein_discriminator_loss(y_real: torch.Tensor, y_fake: torch.Tensor) -> torch.Tensor:
    return -(torch.mean(y_real) - torch.mean(y_fake))


def wasserstein_generator_loss(y_fake: torch.Tensor) -> torch.Tensor:
    return -torch.mean(y_fake)


# if __name__ == '__main__':
#     criterion_sound = VGGishDistance(config['perceptual_loss']['layers_vggish'])
#     criterion_sound_2 = RhoDistance(config['perceptual_loss']['layers_vggish'])
#     sp=torch.zeros((42,128,128))
#     vggish_loss = criterion_sound(sp[:, None, ...], sp[:, None, ...])
    # vggish_loss = criterion_sound_2(sp[:, None, ...], sp[:, None, ...])