### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

### This script was modified based on the pix2pixHD official implementation (see license above)
### https://github.com/NVIDIA/pix2pixHD

import os.path
from data.base_dataset import BaseDataset
from data.data_util import *
import torch
import nibabel as nib
import numpy as np
import random


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        # Override dir_A and dir_B if provided via command line arguments
        if hasattr(opt, 'dir_A') and opt.dir_A != '':
            self.dir_A = opt.dir_A
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + '_A')
            
        if hasattr(opt, 'dir_B') and opt.dir_B != '':
            self.dir_B = opt.dir_B
        else:
            self.dir_B = os.path.join(opt.dataroot, opt.phase + '_B')

        all_A_paths = make_dataset(self.dir_A, opt.extension)
        all_B_paths = make_dataset(self.dir_B, opt.extension)
        
        # Debug prints per aiutare l'utente su Kaggle
        if not all_A_paths:
            print(f"DEBUG: Nessun file trovato in {self.dir_A} con estensione {opt.extension}")
        if not all_B_paths:
            print(f"DEBUG: Nessun file trovato in {self.dir_B} con estensione {opt.extension}")

        if hasattr(opt, 'code_list') and opt.code_list != '':
            # Load codes from the text file
            with open(opt.code_list, 'r') as f:
                codes = [line.strip() for line in f.readlines() if line.strip()]
            
            # --- NUOVO DEBUG ---
            codes_in_A = set([os.path.basename(p).split('_')[0] for p in all_A_paths])
            codes_in_B = set([os.path.basename(p).split('_')[0] for p in all_B_paths])
            target_codes = set(codes)
            
            print(f"DEBUG INFO MATCHER:")
            print(f"- Codici richiesti (dal txt): {len(target_codes)}")
            print(f"- Codici estratti da A: {len(codes_in_A)}")
            print(f"- Codici estratti da B: {len(codes_in_B)}")
            print(f"- Match tra txt e A: {len(target_codes.intersection(codes_in_A))}")
            print(f"- Match tra txt e B: {len(target_codes.intersection(codes_in_B))}")
            print(f"- Esempi txt: {list(target_codes)[:5]}")
            print(f"- Esempi in A: {list(codes_in_A)[:5]}")
            print(f"- Esempi in B: {list(codes_in_B)[:5]}")
            # -------------------

            self.A_paths = []
            self.B_paths = []
            
            for code in codes:
                # Estraiamo il codice esatto dal nome del file: CODICE_SIGLAMRI.nii -> split('_')[0] = CODICE
                matched_A = [p for p in all_A_paths if os.path.basename(p).split('_')[0] == code]
                matched_B = [p for p in all_B_paths if os.path.basename(p).split('_')[0] == code]
                
                if matched_A and matched_B:
                    # Append the first match
                    self.A_paths.append(matched_A[0])
                    self.B_paths.append(matched_B[0])
                else:
                    print(f"Warning: Could not find matching pairs for code {code}")
                    if len(all_A_paths) > 0 and codes.index(code) == 0:
                        print(f"Esempio di file in A: {os.path.basename(all_A_paths[0])} (codice estratto: {os.path.basename(all_A_paths[0]).split('_')[0]})")
                    if len(all_B_paths) > 0 and codes.index(code) == 0:
                        print(f"Esempio di file in B: {os.path.basename(all_B_paths[0])} (codice estratto: {os.path.basename(all_B_paths[0]).split('_')[0]})")
        else:
            self.A_paths = sorted(all_A_paths)
            self.B_paths = sorted(all_B_paths)

        assert self.A_paths, 'modality A can not find files with extension ' + opt.extension
        assert self.B_paths, 'modality B can not find files with extension ' + opt.extension
        assert len(self.A_paths) == len(self.B_paths), 'modality A and B must have the same number of files'

        self.dataset_size = len(self.A_paths)

    def __getitem__(self, index):

        x, y, z = self.opt.patch_size

        ### modality A

        tmp_scansA = np.squeeze(nib.load(self.A_paths[index]).get_fdata())
        tmp_scansB = np.squeeze(nib.load(self.B_paths[index]).get_fdata())
        assert tmp_scansA.shape == tmp_scansB.shape, 'paired scans must have the same shape'

        tmp_scansA[tmp_scansA < 0] = 0
        tmp_scansB[tmp_scansB < 0] = 0
        tmp_scansA = norm_img(tmp_scansA, self.opt.norm_perc)
        tmp_scansB = norm_img(tmp_scansB, self.opt.norm_perc)

        tmp_scansA = torch.unsqueeze(torch.from_numpy(tmp_scansA).float(), 0)
        tmp_scansB = torch.unsqueeze(torch.from_numpy(tmp_scansB).float(), 0)
        _, x1, y1, z1 = tmp_scansA.shape

        if self.opt.dimension.startswith('2'):
            if self.opt.remove_bg:
                bound = get_bounds(tmp_scansA)
            else:
                bound = [0, x1, 0, y1, 0, z1]

            slice_idx = random.sample(range(bound[-2], bound[-1]), 1)[0]

            input_dict = {'img_A': tmp_scansA[:, :, :, slice_idx],
                          'img_B': tmp_scansB[:, :, :, slice_idx]}

        elif self.opt.dimension.startswith('3'):
            for i in range(3):
                assert tmp_scansA.shape[i+1] >= self.opt.patch_size[i], self.A_paths[index] + ' ' + str(
                    i + 1) + ' dimension is smaller than corresponding patch size'

            if self.opt.remove_bg:
                bound = get_bounds(tmp_scansA[0])
                assert bound[1] - x > bound[0], 'first dimension is smaller than patch size after removing background, ' \
                                                'cosider padding or setting remove_bg as false '
                assert bound[3] - y > bound[
                    2], 'second dimension is smaller than patch size after removing background, ' \
                        'cosider padding or setting remove_bg as false '
                assert bound[5] - z > bound[4], 'third dimension is smaller than patch size after removing background, ' \
                                                'cosider padding or setting remove_bg as false '

                x_idx = random.sample(range(bound[0], bound[1] - x), 1)[0]
                y_idx = random.sample(range(bound[2], bound[3] - y), 1)[0]
                z_idx = random.sample(range(bound[4], bound[5] - z), 1)[0]
            else:
                bound = [0, x1, 0, y1, 0, z1]
                if x1 - x == 0:
                    x_idx = 0
                else:
                    x_idx = random.sample(range(bound[0], bound[1] - x), 1)[0]
                if y1 - y == 0:
                    y_idx = 0
                else:
                    y_idx = random.sample(range(bound[2], bound[3] - y), 1)[0]
                if z1 - z == 0:
                    z_idx = 0
                else:
                    z_idx = random.sample(range(bound[4], bound[5] - z), 1)[0]
            input_dict = {'img_A': tmp_scansA[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z],
                          'img_B': tmp_scansB[:, x_idx:x_idx + x, y_idx:y_idx + y, z_idx:z_idx + z]}

        return input_dict

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'Paired/Aligned Dataset'
