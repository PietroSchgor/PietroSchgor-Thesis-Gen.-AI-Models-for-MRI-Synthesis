### This code is largely borrowed from pix2pixHD pytorch implementation
### https://github.com/NVIDIA/pix2pixHD

import time
import os
import numpy as np
import torch
from torch.autograd import Variable
from collections import OrderedDict
import math
import copy
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import random
import lpips
from validation_metrics import FSIM, calculate_3d_full_ssim

# Ottimizzazione CUDNN: seleziona l'algoritmo di convoluzione più veloce
# (Perfetto dato che la nostra patch_size è sempre fissa a 64x64x64)
cudnn.benchmark = True

from models.models import create_model
import torch.nn as nn
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer
from util.image_pool import ImagePool
from models.networks import GANLoss, feature_loss, discriminate
def lcm(a, b): return abs(a * b) / math.gcd(a, b) if a and b else 0

##############################################################################
# Initialize options
##############################################################################

opt = TrainOptions().parse()
iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
start_epoch, epoch_iter = 1, 0

# Caricamento epoche in caso di resume
load_iter_path = opt.iter_file if hasattr(opt, 'iter_file') and opt.iter_file != '' else iter_path
if hasattr(opt, 'resume_G') and opt.resume_G != '' and os.path.exists(load_iter_path):
    try:
        iter_data = np.loadtxt(load_iter_path, delimiter=',')
        start_epoch = int(iter_data[0])
        epoch_iter = int(iter_data[1])
        print(f"Resuming from epoch {start_epoch}, iter {epoch_iter} from {load_iter_path}")
    except Exception as e:
        print(f"Could not load iter state from {load_iter_path}: {e}")

opt.print_freq = lcm(opt.print_freq, opt.batchSize)
if opt.debug:
    opt.display_freq = 1
    opt.print_freq = 1
    opt.niter = 1
    opt.niter_decay = 0
    opt.max_dataset_size = 10
ler = opt.lr

##############################################################################
# Initialize dataloader
##############################################################################

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# Setup TensorBoard
writer = SummaryWriter(os.path.join(opt.checkpoints_dir, opt.name, 'runs'))

val_dataset_size = 0
if hasattr(opt, 'val_code_list') and opt.val_code_list != '':
    val_opt = copy.deepcopy(opt)
    val_opt.phase = 'val'
    val_opt.code_list = opt.val_code_list
    if hasattr(opt, 'val_dir_A') and opt.val_dir_A != '':
        val_opt.dir_A = opt.val_dir_A
    if hasattr(opt, 'val_dir_B') and opt.val_dir_B != '':
        val_opt.dir_B = opt.val_dir_B
    val_data_loader = CreateDataLoader(val_opt)
    val_dataset = val_data_loader.load_data()
    val_dataset_size = len(val_data_loader)
    print('#validation images = %d' % val_dataset_size)

best_val_loss = float('inf')
patience_counter = 0

# Path di salvataggio nella nuova sessione
es_path = os.path.join(opt.checkpoints_dir, opt.name, 'early_stopping.txt')

# Path di caricamento: usa quello passato da terminale, altrimenti usa es_path
load_es_path = opt.early_stopping_file if hasattr(opt, 'early_stopping_file') and opt.early_stopping_file != '' else es_path

if hasattr(opt, 'resume_G') and opt.resume_G != '' and os.path.exists(load_es_path):
    try:
        es_data = np.loadtxt(load_es_path, delimiter=',')
        best_val_loss = float(es_data[0])
        patience_counter = int(es_data[1])
        print(f"Resuming early stopping state from {load_es_path}: best_val_loss={best_val_loss:.4f}, patience={patience_counter}")
    except Exception as e:
        print(f"Could not load early stopping state from {load_es_path}: {e}")

##############################################################################
# Initialize networks
##############################################################################

PTNet, D, ext_discriminator = create_model(opt)

if hasattr(opt, 'resume_G') and opt.resume_G != '':
    print(f'Loading PTNet (Generator) from {opt.resume_G}...')
    PTNet.load_state_dict(torch.load(opt.resume_G))

if hasattr(opt, 'resume_D') and opt.resume_D != '':
    print(f'Loading D (Discriminator) from {opt.resume_D}...')
    D.load_state_dict(torch.load(opt.resume_D))

PTNet.cuda()
D.cuda()
ext_discriminator.cuda()
ext_discriminator.eval()
for param in ext_discriminator.parameters():
    param.requires_grad = False

##############################################################################
# Initialize util components
##############################################################################

optimizer_PTNet = torch.optim.Adam(PTNet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0)
optimizer_D = torch.optim.Adam(D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
fake_pool = ImagePool(0)

CE = nn.CrossEntropyLoss()
mse = nn.MSELoss()

if hasattr(opt, 'val_code_list') and opt.val_code_list != '':
    print("Inizializzazione modelli percettivi LPIPS e FSIM per la validazione...")
    lpips_model = lpips.LPIPS(net='alex', verbose=False).cuda()
    lpips_model.eval()
    fsim_model = FSIM().cuda()
    fsim_model.eval()
criterionGAN = GANLoss(use_lsgan=not False, tensor=torch.cuda.FloatTensor)

# training/display parameter
visualizer = Visualizer(opt)
total_steps = (start_epoch - 1) * dataset_size + epoch_iter
display_delta = total_steps % opt.display_freq
print_delta = total_steps % opt.print_freq
save_delta = total_steps % opt.save_latest_freq

##############################################################################
# Training code
##############################################################################

# scaler rimosso per disabilitare AMP

for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    if epoch != start_epoch:
        epoch_iter = epoch_iter % dataset_size
    for i, data in enumerate(dataset, start=epoch_iter):
        if total_steps % opt.print_freq == print_delta:
            iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize

        # whether to collect output images
        save_fake = total_steps % opt.display_freq == display_delta

        ##############################################################################
        # Forward Pass
        ##############################################################################
        
        input_image = Variable(data['img_A'].cuda(non_blocking=True))
        target_image = Variable(data['img_B'].cuda(non_blocking=True))

        # Synthesize and MSE loss
        generated = PTNet(input_image)
        loss_mse = mse(generated, target_image)

        # Fake Detection and Loss
        pred_fake_pool = discriminate(D, fake_pool, input_image, generated, use_pool=True)
        loss_D_fake = criterionGAN(pred_fake_pool, False)

        # Real Detection and Loss
        pred_real = discriminate(D, fake_pool, input_image, target_image)
        loss_D_real = criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)
        pred_fake = D.forward(torch.cat((input_image, generated), dim=1))
        loss_G_GAN = criterionGAN(pred_fake, True)

        # GAN feature matching loss
        loss_G_GAN_Feat, loss_G_GAN_Feat_ext = feature_loss(opt, target_image, generated, pred_real, pred_fake,
                                                            ext_discriminator)

        # Compute overall loss
        loss_D = (loss_D_fake + loss_D_real) * 0.5
        loss_G = loss_mse * 100.0 + loss_G_GAN + loss_G_GAN_Feat_ext * 10.0 + loss_G_GAN_Feat * 10.0
        
        loss_dict = dict(
            zip(['MSE', 'G_GAN', 'G_GAN_Feat_ext', 'G_GAN_Feat', 'D_fake', 'D_real'], [loss_mse.item(),
                                                                                       loss_G_GAN.item(),
                                                                                       loss_G_GAN_Feat_ext.item(),
                                                                                       loss_G_GAN_Feat.item(),
                                                                                       loss_D_fake.item(),
                                                                                       loss_D_real.item()]), )
        ##############################################################################
        # Backward Pass
        ##############################################################################

        # update generator weights
        optimizer_PTNet.zero_grad()
        loss_G.backward()
        optimizer_PTNet.step()

        # update discriminator weights
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        ##############################################################################
        # Display results, print out loss, and save latest model
        ##############################################################################

        # print out loss
        if total_steps % opt.print_freq == print_delta:
            errors = {k: v if not isinstance(v, int) else v for k, v in loss_dict.items()}
            t = (time.time() - iter_start_time) / opt.print_freq
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            for k, v in errors.items():
                writer.add_scalar(f'Train/{k}', v, total_steps)

        # display output images
        if save_fake:
            visuals = OrderedDict([('input_image', util.tensor2im(data['img_A'][0, :, :, :, 15])),
                                   ('synthesized_image', util.tensor2im(generated.data[0, :, :, :, 15])),
                                   ('target_image', util.tensor2im(data['img_B'][0, :, :, :, 15]))])
            visualizer.display_current_results(visuals, epoch, total_steps)

        # save latest model
        if total_steps % opt.save_latest_freq == save_delta:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            torch.save(PTNet.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_latest.pth'))
            np.savetxt(iter_path, (epoch, epoch_iter), delimiter=',', fmt='%d')
            torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_latest.pth'))

        if epoch_iter >= dataset_size:
            break

    # end of epoch
    iter_end_time = time.time()
    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    # Validation Loop
    if hasattr(opt, 'val_code_list') and opt.val_code_list != '' and val_dataset_size > 0:
        PTNet.eval()
        val_lpips_list, val_fsim_list, val_ssim_list = [], [], []
        
        with torch.no_grad():
            for val_data in val_dataset:
                val_input = Variable(val_data['img_A'].cuda(non_blocking=True))
                val_target = Variable(val_data['img_B'].cuda(non_blocking=True))
                val_generated = PTNet(val_input)
                
                batch_size = val_generated.size(0)
                for b in range(batch_size):
                    # 1. SSIM su NumPy 3D (H, W, D)
                    r_np = val_target[b, 0].cpu().numpy()
                    f_np = val_generated[b, 0].cpu().numpy()
                    r_np = np.transpose(r_np, (1, 2, 0))
                    f_np = np.transpose(f_np, (1, 2, 0))
                    
                    s_mean, _, _ = calculate_3d_full_ssim(r_np, f_np)
                    val_ssim_list.append(s_mean)
                    
                    # 2. LPIPS e FSIM su subset random di slice
                    D_dim = val_generated.size(2)
                    num_slices_perc = min(20, D_dim) # Estrai fino a 20 slice random
                    slice_indices = random.sample(range(D_dim), num_slices_perc)
                    
                    # Forma: (num_slices, 1, H, W)
                    real_slices = val_target[b:b+1, :, slice_indices, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 1, val_generated.size(3), val_generated.size(4))
                    fake_slices = val_generated[b:b+1, :, slice_indices, :, :].permute(0, 2, 1, 3, 4).reshape(-1, 1, val_generated.size(3), val_generated.size(4))
                    
                    # LPIPS: 3 canali [-1, 1]
                    real_rgb = real_slices.repeat(1, 3, 1, 1)
                    fake_rgb = fake_slices.repeat(1, 3, 1, 1)
                    lpips_score = lpips_model(fake_rgb, real_rgb).mean().item()
                    val_lpips_list.append(lpips_score)
                    
                    # FSIM: range [0, 255]
                    real_255 = ((real_slices + 1.0) / 2.0) * 255.0
                    fake_255 = ((fake_slices + 1.0) / 2.0) * 255.0
                    fsim_score = fsim_model(fake_255, real_255).item()
                    val_fsim_list.append(fsim_score)
        
        avg_lpips = np.mean(val_lpips_list)
        std_lpips = np.std(val_lpips_list)
        avg_fsim = np.mean(val_fsim_list)
        std_fsim = np.std(val_fsim_list)
        avg_ssim = np.mean(val_ssim_list)
        std_ssim = np.std(val_ssim_list)
        
        # Calcolo Loss Composita per Early Stopping
        val_loss = opt.weight_lpips * avg_lpips + opt.weight_fsim * (1.0 - avg_fsim) + opt.weight_ssim * (1.0 - avg_ssim)
        
        print(f"Epoch {epoch} - Validation Metrics:")
        print(f"  LPIPS: {avg_lpips:.4f} ± {std_lpips:.4f} (peso {opt.weight_lpips:.1f}, ↓ meglio)")
        print(f"  FSIM:  {avg_fsim:.4f} ± {std_fsim:.4f} (peso {opt.weight_fsim:.1f}, ↑ meglio)")
        print(f"  SSIM:  {avg_ssim:.4f} ± {std_ssim:.4f} (peso {opt.weight_ssim:.1f}, ↑ meglio)")
        print(f"  Composite Loss: {val_loss:.4f}")
        
        writer.add_scalar('Val/Loss_Weighted', val_loss, epoch)
        writer.add_scalar('Val/LPIPS', avg_lpips, epoch)
        writer.add_scalar('Val/LPIPS_std', std_lpips, epoch)
        writer.add_scalar('Val/FSIM', avg_fsim, epoch)
        writer.add_scalar('Val/FSIM_std', std_fsim, epoch)
        writer.add_scalar('Val/SSIM', avg_ssim, epoch)
        writer.add_scalar('Val/SSIM_std', std_ssim, epoch)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'Validation loss migliorata a {best_val_loss:.4f}. Salvataggio miglior modello...')
            torch.save(PTNet.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_best.pth'))
        else:
            patience_counter += 1
            print(f'Nessun miglioramento. Pazienza: {patience_counter}/{opt.patience}')
            
        # Salva lo stato dell'early stopping
        np.savetxt(es_path, (best_val_loss, patience_counter), delimiter=',', fmt='%f')
            
        PTNet.train()
        
        if patience_counter >= opt.patience:
            print(f'Early stopping attivato all\'epoca {epoch}!')
            break

    # save model for this epoch
    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        torch.save(PTNet.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_ckpt%d%d.pth' % (epoch, total_steps)))
        np.savetxt(iter_path, (epoch + 1, 0), delimiter=',', fmt='%d')
        torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_ckpt%d%d.pth' % (epoch, total_steps)))

    # linearly decay learning rate after certain iterations
    if epoch > opt.niter:
        ler -= (opt.lr) / (opt.niter_decay)
        for param_group in optimizer_PTNet.param_groups:
            param_group['lr'] = ler
            print('change lr to ')
            print(param_group['lr'])
        for param_group in optimizer_D.param_groups:
            param_group['lr'] = ler

print('Training finished! Saving final model...')
writer.close()
torch.save(PTNet.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'PTNet_final.pth'))
torch.save(D.state_dict(), os.path.join(opt.checkpoints_dir, opt.name, 'D_final.pth'))
print('Final model saved in', os.path.join(opt.checkpoints_dir, opt.name))
