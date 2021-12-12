import os
import itertools
import pickle
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

import wandb

from model.modules import LatentModel, AmortizedModel,VGGDistance,VGGishDistance,\
constractiveDistance,RhoDistance,Discriminator,wasserstein_discriminator_loss,\
wasserstein_generator_loss,SpectralConvergenceLoss,reconstraction_criteria
from model.utils import AverageMeter, NamedTensorDataset
from wav2mel import Wav2Mel
import torchaudio.transforms as T

class Lord:

	def __init__(self, config=None):
		super().__init__()
		self.config = config
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.latent_model = None
		self.discriminator=None
		self.amortized_model = None
		self.reconstraction_criterion=None
		self.optimizer_d=None
		self.scheduler=None
		self.optimizer_g=None

	def load(self, model_dir, latent=True, amortized=True):
		with open(os.path.join(model_dir, 'config.pkl'), 'rb') as config_fd:
			self.config = pickle.load(config_fd)

		if latent:
			self.latent_model = LatentModel(self.config)
			self.latent_model.load_state_dict(torch.load(os.path.join(model_dir, 'latent.pth')))

		if amortized:
			self.amortized_model = AmortizedModel(self.config)
			self.amortized_model.load_state_dict(torch.load(os.path.join(model_dir, 'amortized.pth')))

	def save(self, model_dir, latent=True, amortized=True):
		with open(os.path.join(model_dir, 'config.pkl'), 'wb') as config_fd:
			pickle.dump(self.config, config_fd)

		if latent:
			torch.save(self.latent_model.state_dict(), os.path.join(model_dir, 'latent.pth'))

		if amortized:
			torch.save(self.amortized_model.state_dict(), os.path.join(model_dir, 'amortized.pth'))

	def train_discreminator(self,batch,train=True):
		self.optimizer_d.zero_grad(set_to_none=True)
		genrated_Data= self.amortized_model(batch['img'])
		real_disc_predictions = self.discriminator(batch["img"][:, None, ...])
		fake_disc_predictions = self.discriminator(genrated_Data["img"][:, None, ...])
		disc_loss = wasserstein_discriminator_loss(real_disc_predictions, fake_disc_predictions)\
					+self.discriminator.gradient_penalty(batch["img"][:, None, ...],genrated_Data["img"][:, None, ...])
		if train:
			disc_loss.backward()
			self.optimizer_d.step()
		return disc_loss

	def train_decoder(self,batch,train=True,adverarial=False,content_decay=True,model="latent"):
		self.optimizer_g.zero_grad(set_to_none=True)
		if model=="latent":
			out = self.latent_model(batch['img_id'], batch['class_id'])
		elif model=="ammortized":
			out=self.amortized_model(batch['img'])
		losses=dict()
		if content_decay:
			content_penalty = torch.sum(out['content_code'] ** 2, dim=1).mean()
			loss=self.config['content_decay'] * content_penalty
			losses["content"]=content_penalty
		else:
			loss=0
		mse = ( torch.nn.functional.l1_loss(out['img'], batch['img'][:, None, ...]))
		losses["mse"]=mse
		rec_loss,rec_loss_log=self.reconstraction_criterion(out['img'], batch['img'])
		loss+=rec_loss
		losses.update(rec_loss_log)
		if adverarial:
			fake_disc_predictions = self.discriminator(out["img"][:, None, ...])
			gan_loss = -torch.mean(fake_disc_predictions)*self.config['loss_weights']["adv_loss"]
			losses["gan_loss"]=gan_loss
			loss+=gan_loss
		if train:
			loss.backward()
			self.optimizer_g.step()
			self.scheduler.step()
		losses["loss"]=loss
		return losses


	def train_latent(self, imgs, classes, model_dir):
		self.latent_model = LatentModel(self.config)
		data = dict(
			img=torch.from_numpy(imgs),
			img_id=torch.arange(imgs.shape[0]).type(torch.int64),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.latent_model.init()
		self.latent_model.to(self.device)
		self.reconstraction_criterion=reconstraction_criteria(self.config).to(self.device)

		self.optimizer_g = Adam([
			{
				'params': self.latent_model.decoder.parameters(),
				'lr': self.config['train']['learning_rate']['generator']
			},
			{
				'params': itertools.chain(self.latent_model.content_embedding.parameters(), self.latent_model.class_embedding.parameters()),
				'lr': self.config['train']['learning_rate']['latent']
			}
		], betas=(0.5, 0.999))



		self.scheduler = CosineAnnealingLR(
			self.optimizer_g,
			T_max=self.config['train']['n_epochs'] * len(data_loader),
			eta_min=self.config['train']['learning_rate']['min']
		)

		visualized_imgs = []
		train_loss = AverageMeter()
		image_loss=AverageMeter()
		sound_loss=AverageMeter()
		sp_loss=AverageMeter()
		mse_loss=AverageMeter()
		sparsity_loss=AverageMeter()
		rec_loss=AverageMeter()
		content_loss=AverageMeter()
		with torch.autograd.set_detect_anomaly(True):

			for epoch in range(self.config['train']['n_epochs']):
				self.latent_model.train()
				# discriminator.train()
				train_loss.reset()
				image_loss.reset()
				sound_loss.reset()
				sp_loss.reset()
				rec_loss.reset()
				content_loss.reset()

				pbar = tqdm(iterable=data_loader,miniters=100)
				for i,batch in enumerate(pbar):
					batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
					# if epoch>70:
					# 	disc_loss=self.train_discreminator(batch,train=(i%5==0))
					# 	disc_meter.update(disc_loss.item())
					losses=self.train_decoder(batch,adverarial=False)
					train_loss.update(losses["loss"].item())
					mse_loss.update(losses["mse"].item())
					content_loss.update(losses["content"].item())
					if "Rho" in self.config["loss_types"] or "Vggish" in self.config["loss_types"]:
						sound_loss.update(losses["sound_loss"].item())
					if "image" in self.config["loss_types"]:
						image_loss.update(losses["image_loss"].item())
					if "sp_loss" in self.config["loss_types"]:
						sp_loss.update(losses["sp_loss"].item())
					if "sparsity" in self.config["loss_types"]:
						sparsity_loss.update(losses["sparsity_loss"].item())
					rec_loss.update(losses["rec_loss"].item())

					pbar.set_description_str('epoch #{}'.format(epoch))
					pbar.set_postfix(loss=train_loss.avg)

				pbar.close()
				self.save(model_dir, latent=True, amortized=False)

				wandb.log({
					'loss': train_loss.avg,
					'content_penalty':content_loss.avg,
					'fro_norm':sp_loss.avg,
					'image_loss':image_loss.avg,
					'sp_loss':sp_loss.avg,
					'sound_loss': sound_loss.avg,
					'mse':mse_loss.avg,
					'sparsity_loss':sparsity_loss.avg,
					'decoder_lr': self.scheduler.get_last_lr()[0],
					'latent_lr': self.scheduler.get_last_lr()[1],
				}, step=epoch)

				with torch.no_grad():
					fixed_sample_img = self.generate_samples(dataset, step=epoch)
				#
				wandb.log({f'generated-{epoch}': [wandb.Image(fixed_sample_img)]}, step=epoch)
				visualized_imgs.append(np.asarray(fixed_sample_img).transpose(2,0,1)[:3])

				if epoch % 5 == 0:
					wandb.log({f'video': [
						wandb.Video(np.array(visualized_imgs)),
					]}, step=epoch)
		self.save("models",latent=True,amortized=False)

	def train_amortized(self, imgs, classes, model_dir):
		self.amortized_model = AmortizedModel(self.config)
		wandb.config.update(self.config)
		self.amortized_model.decoder.load_state_dict(self.latent_model.decoder.state_dict())

		data = dict(
			img=torch.from_numpy(imgs),
			img_id=torch.arange(imgs.shape[0]).type(torch.int64),
			class_id=torch.from_numpy(classes.astype(np.int64))
		)

		dataset = NamedTensorDataset(data)
		data_loader = DataLoader(
			dataset, batch_size=self.config['train_encoders']['batch_size'],
			shuffle=True, sampler=None, batch_sampler=None,
			num_workers=1, pin_memory=True, drop_last=True
		)

		self.latent_model.to(self.device)
		self.amortized_model.to(self.device)
		self.discriminator=Discriminator(1).to(self.device)
		self.reconstraction_criterion=reconstraction_criteria(self.config).to(self.device)
		embedding_criterion = nn.MSELoss()

		self.optimizer_g= Adam(
			params=self.amortized_model.parameters(),
			lr=self.config['train_encoders']['learning_rate']['max'],
			betas=(0.9, 0.999)
		)

		self.scheduler= CosineAnnealingLR(
			self.optimizer_g,
			T_max=self.config['train_encoders']['n_epochs'] * len(data_loader),
			eta_min=self.config['train_encoders']['learning_rate']['min']
		)
		self.optimizer_d = Adam([
			{
				'params': self.discriminator.parameters(),
				'lr': self.config['train']['learning_rate']['disc']
			},
		], betas=(0.9, 0.999))
		
		visualized_imgs = []

		train_loss_meter = AverageMeter()
		rec_loss_meter=AverageMeter()
		content_loss_meter=AverageMeter()
		class_loss_meter=AverageMeter()
		image_loss_meter=AverageMeter()
		gan_loss_meter=AverageMeter()
		disc_loss_meter=AverageMeter()

		for epoch in range(self.config['train_encoders']['n_epochs']):
			train_loss_meter.reset()
			image_loss_meter.reset()
			image_loss_meter.reset()
			class_loss_meter.reset()
			content_loss_meter.reset()

			self.latent_model.eval()
			self.amortized_model.train()

			pbar = tqdm(iterable=data_loader)
			for i,batch in enumerate(pbar):
				batch = {name: tensor.to(self.device) for name, tensor in batch.items()}
				self.optimizer_g.zero_grad(set_to_none=True)
				target_content_code = self.latent_model.content_embedding(batch['img_id'])
				target_class_code = self.latent_model.class_embedding(batch['class_id'])
				target_content_code= target_content_code.reshape((-1, 128, 8))
				out = self.amortized_model(batch['img'])
				rec_losses = self.train_decoder(batch, adverarial=(epoch>12),train=False,content_decay=False,model="ammortized")
				loss_reconstruction = rec_losses["image_loss"]+rec_losses["sound_loss"]
				loss_content = 12*embedding_criterion(out['content_code'].reshape(target_content_code.shape), target_content_code)
				loss_class = 12*embedding_criterion(out['class_code'], target_class_code)
				loss = loss_reconstruction +  loss_content + loss_class
				if epoch>12:
					loss+=rec_losses["gan_loss"]
					gan_loss_meter.update(rec_losses["gan_loss"].item())
					loss.backward()
					self.optimizer_g.step()
					self.scheduler.step()
					if i%5==0:
						disc_loss=self.train_discreminator(batch)
						disc_loss_meter.update(disc_loss.item())
				else:
					loss.backward()
					self.optimizer_g.step()
					self.scheduler.step()


				pbar.set_description_str('epoch #{}'.format(epoch))
				pbar.set_postfix(loss=train_loss_meter.avg)
				train_loss_meter.update(loss.item())
				class_loss_meter.update(loss_class.item())
				content_loss_meter.update(loss_content.item())
				rec_loss_meter.update(loss_reconstruction.item())
				image_loss_meter.update(rec_losses["image_loss"].item())

			pbar.close()
			self.save(model_dir, latent=False, amortized=True)

			wandb.log({
				'loss-amortized': train_loss_meter.avg,
				'rec-loss-amortized': rec_loss_meter.avg,
				'content-loss-amortized': class_loss_meter.avg,
				'class-loss-amortized': content_loss_meter.avg,
				"image_loss":image_loss_meter.avg,
				"gan_loss":gan_loss_meter.avg,
				"disc_loss":disc_loss_meter.avg
			}, step=epoch)

			with torch.no_grad():
				fixed_sample_img = self.generate_samples_amortized(dataset, step=epoch)

			wandb.log({f'generated-{epoch}': [wandb.Image(fixed_sample_img)]}, step=epoch)
			visualized_imgs.append(np.asarray(fixed_sample_img).transpose(2,0,1)[:3])

			if epoch % 5 == 0:
				wandb.log({f'video': [
					wandb.Video(np.array(visualized_imgs)),
				]}, step=epoch)


	@staticmethod
	def random_from_intervals(intervals):
		out=[]
		for i in range(1,len(intervals)):
			np.random.RandomState(seed=1234).choice(len(dataset), size=n_samples, replace=False).astype(np.int64)

	def generate_samples(self, dataset, n_samples=4, step=None):
		self.latent_model.eval()

		img_idx=self.config["samples"]
		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
		fig = plt.figure(figsize=(10, 10))
		if step:
			fig.suptitle(f'Step={step}')
		for i in range(n_samples):
			# Plot row headers - instruments
			plt.subplot(n_samples + 1, n_samples + 1,
						n_samples + 1 + i * (n_samples + 1) + 1)
			plt.imshow(samples['img'][i].detach().cpu().numpy(), cmap='inferno')
			plt.gca().invert_yaxis()
			plt.axis('off')

			# Plot column headers -content
			plt.subplot(n_samples + 1, n_samples + 1, i + 2)
			plt.imshow(samples['img'][i].detach().cpu().numpy(), cmap='inferno')
			plt.gca().invert_yaxis()
			plt.axis('off')

			for j in range(n_samples):
				plt.subplot(n_samples + 1, n_samples + 1,
							n_samples + 2 + i * (n_samples + 1) + j + 1)

				content_id = samples['img_id'][[j]]
				class_id = samples['class_id'][[i]]
				cvt = self.latent_model(content_id, class_id)['img'][0].detach().cpu().numpy()

				if step % 5 == 0:
					np.savez(f'samples_lib/{step}_{content_id.item()}({samples["class_id"][[j]].item()})to{class_id.item()}.npz', cvt)
				cvt= cvt if len(cvt.shape)==2 else cvt[0]
				plt.imshow(cvt, cmap='inferno')
				plt.gca().invert_yaxis()
				plt.axis('off')

		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		pil_img = Image.open(buf)
		return pil_img

	def generate_samples_amortized(self, dataset, n_samples=4, step=None):
		self.amortized_model.eval()

		img_idx=self.config["samples"]
		samples = dataset[img_idx]
		samples = {name: tensor.to(self.device) for name, tensor in samples.items()}
		fig = plt.figure(figsize=(10, 10))
		if step:
			fig.suptitle(f'Step={step}')
		for i in range(n_samples):
			# Plot row headers (speaker)
			plt.subplot(n_samples + 1, n_samples + 1,
						n_samples + 1 + i * (n_samples + 1) + 1)
			plt.imshow(samples['img'][i].detach().cpu().numpy(), cmap='inferno')
			plt.gca().invert_yaxis()
			plt.axis('off')

			# Plot column headers (content)
			plt.subplot(n_samples + 1, n_samples + 1, i + 2)
			plt.imshow(samples['img'][i].detach().cpu().numpy(), cmap='inferno')
			plt.gca().invert_yaxis()
			plt.axis('off')

			for j in range(n_samples):
				plt.subplot(n_samples + 1, n_samples + 1,
							n_samples + 2 + i * (n_samples + 1) + j + 1)

				content_img = samples['img'][[j]]
				class_img = samples['img'][[i]]
				cvt = self.amortized_model.convert(content_img, class_img)['img'][0].detach().cpu().numpy()

				if step % 5 == 0:
					np.savez(f'samples/e{step}_{samples["img_id"][[j]].item()}({samples["class_id"][[j]].item()})to{samples["class_id"][[i]].item()}.npz', cvt)

				plt.imshow(cvt, cmap='inferno')
				plt.gca().invert_yaxis()
				plt.axis('off')

		buf = io.BytesIO()
		plt.savefig(buf, format='png')
		buf.seek(0)
		pil_img = Image.open(buf)
		return pil_img
