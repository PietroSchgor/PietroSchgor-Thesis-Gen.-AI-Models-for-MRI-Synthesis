from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        self.parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        self.parser.add_argument('--resume_G', type=str, default='', help='path to a pretrained Generator (PTNet) to resume training')
        self.parser.add_argument('--resume_D', type=str, default='', help='path to a pretrained Discriminator (D) to resume training')

        # for training

        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--val_code_list', type=str, default='', help='path to text file containing common codes for pairing validation data')
        self.parser.add_argument('--val_dir_A', type=str, default='', help='specific path for validation domain A')
        self.parser.add_argument('--val_dir_B', type=str, default='', help='specific path for validation domain B')
        self.parser.add_argument('--patience', type=int, default=50, help='number of epochs to wait without improvement before stopping training early')
        self.parser.add_argument('--early_stopping_file', type=str, default='', help='path to a previous early_stopping.txt file to resume patience state')
        self.parser.add_argument('--iter_file', type=str, default='', help='path to a previous iter.txt file to resume epoch and iteration count')
        self.parser.add_argument('--weight_lpips', type=float, default=1.0, help='weight for LPIPS metric in early stopping')
        self.parser.add_argument('--weight_fsim', type=float, default=1.0, help='weight for FSIM metric in early stopping')
        self.parser.add_argument('--weight_ssim', type=float, default=1.0, help='weight for SSIM metric in early stopping')
        self.parser.add_argument('--niter', type=int, default=1000, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=1000, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.isTrain = True
