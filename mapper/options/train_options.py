from argparse import ArgumentParser


class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
		self.parser.add_argument('--exp_dir', type=str, help='Path to experiment output directory')
		self.parser.add_argument('--mapper_type', default='LevelsMapper', type=str, help='Which mapper to use')
		self.parser.add_argument('--no_coarse_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_medium_mapper', default=False, action="store_true")
		self.parser.add_argument('--no_fine_mapper', default=False, action="store_true")
		self.parser.add_argument('--latents_train_path', default="../latent_codes/train_faces.pt", type=str, help="The latents for the training")
		self.parser.add_argument('--latents_test_path', default="../latent_codes/test_faces.pt", type=str, help="The latents for the validation")
		self.parser.add_argument('--train_dataset_size', default=5000, type=int, help="Will be used only if no latents are given")
		self.parser.add_argument('--test_dataset_size', default=1000, type=int, help="Will be used only if no latents are given")
		self.parser.add_argument('--w_space', default=False, action="store_true", help="if the input latent code is in w space, i.e. 1*512")

		self.parser.add_argument('--batch_size', default=2, type=int, help='Batch size for training')
		self.parser.add_argument('--test_batch_size', default=1, type=int, help='Batch size for testing and inference')
		self.parser.add_argument('--workers', default=4, type=int, help='Number of train dataloader workers')
		self.parser.add_argument('--test_workers', default=2, type=int, help='Number of test/inference dataloader workers')

		self.parser.add_argument('--learning_rate', default=0.5, type=float, help='Optimizer learning rate')
		self.parser.add_argument('--optim_name', default='ranger', type=str, help='Which optimizer to use')

		self.parser.add_argument('--id_lambda', default=0.2, type=float, help='ID loss multiplier factor')
		self.parser.add_argument('--perc_lambda', default=0, type=float, help='perc loss multiplier factor, default setting uses 0.01')
		self.parser.add_argument('--nce_lambda', default=0.3, type=float, help='CLIP-NCE loss multiplier factor')
		self.parser.add_argument('--latent_l2_lambda', default=0.8, type=float, help='Latent L2 loss multiplier factor')

		self.parser.add_argument('--stylegan_weights', default='../pretrained/stylegan2-ffhq-config-f.pt', type=str, help='Path to StyleGAN model weights')
		self.parser.add_argument('--stylegan_size', default=1024, type=int)
		self.parser.add_argument('--ir_se50_weights', default='../pretrained/model_ir_se50.pth', type=str, help="Path to facial recognition network used in ID loss")
		self.parser.add_argument('--vgg_weights', default='../pretrained/vgg19_conv.pth', type=str, help="Path to VGG network used in perc loss")
		self.parser.add_argument('--checkpoint_path', default=None, type=str, help='Path to StyleCLIPModel model checkpoint')

		self.parser.add_argument('--max_steps', default=50000, type=int, help='Maximum number of training steps')
		self.parser.add_argument('--image_interval', default=500, type=int, help='Interval for logging train images during training')
		self.parser.add_argument('--board_interval', default=50, type=int, help='Interval for logging metrics to tensorboard')
		self.parser.add_argument('--val_interval', default=5000, type=int, help='Validation interval')
		self.parser.add_argument('--save_interval', default=5000, type=int, help='Model checkpoint interval')
		self.parser.add_argument('--source_text', default='face', type=str, help='source text prompt')	
		self.parser.add_argument('--description', required=True, type=str, help='Driving text prompt')


	def parse(self):
		opts = self.parser.parse_args()
		return opts