import resource
import os
import random

import torch
import torch.utils.data as Data

from config import opt
from utils import non_model
from make_dataset import test_Dataset
from net import model_TransSR

import numpy as np

from tqdm import tqdm
import SimpleITK as sitk
import warnings
warnings.filterwarnings("ignore")

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2000, rlimit[1]))

def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        value = value.strip().lower()
        if value in ['1', 'true', 'yes', 'y', 'on']:
            return True
        if value in ['0', 'false', 'no', 'n', 'off']:
            return False
    return bool(value)


def test(**kwargs):
    # stage 1
    kwargs, data_info_dict = non_model.read_kwargs(kwargs)
    compute_ssim = _to_bool(kwargs.pop('compute_ssim', True))
    ssim_batch_size = int(kwargs.pop('ssim_batch_size', 32))
    ssim_stride = int(kwargs.pop('ssim_stride', 1))
    if ssim_batch_size <= 0:
        raise ValueError('ssim_batch_size must be > 0')
    if ssim_stride <= 0:
        raise ValueError('ssim_stride must be > 0')

    opt.load_config('../config/default.txt')
    config_dict = opt._spec(kwargs)

    # stage 2
    save_model_folder = '../model/%s/%s/' % (opt.path_key, str(opt.net_idx))
    save_output_folder = '../test_output/%s/%s/' % (
        opt.path_key, str(opt.net_idx))
    os.makedirs(save_output_folder, exist_ok=True)

    # stage 3
    save_model_list = sorted(os.listdir(save_model_folder))
    use_model = [each for each in save_model_list if each.endswith('pkl')][0]
    use_model_path = save_model_folder + use_model
    config_dict = non_model.update_kwargs(use_model_path, kwargs)
    opt._spec(config_dict)
    print('load config done')

    # stage 4 Dataloader Setting
    def set_seed(seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    GLOBAL_WORKER_ID = None

    def worker_init_fn(worker_id):
        global GLOBAL_WORKER_ID
        GLOBAL_WORKER_ID = worker_id
        set_seed(GLOBAL_SEED + worker_id)

    GLOBAL_SEED = 2022
    non_model.seed_everything(GLOBAL_SEED)

    ###### Device ######
    device = non_model.resolve_device(opt.gpu_idx)
    print('Use device:', device)
    if compute_ssim:
        print(f'SSIM enabled, batch_size={ssim_batch_size}, stride={ssim_stride}')
    else:
        print('SSIM disabled by compute_ssim=False')

    save_model_path = save_model_folder + use_model
    save_dict = torch.load(save_model_path, map_location=torch.device('cpu'))
    config_dict = save_dict['config_dict']
    config_dict.pop('path_img')
    config_dict['mode'] = 'test'
    opt._spec(config_dict)

    # test set
    test_list, test_only_1mm, test_only_5mm = non_model.list_paired_cases(opt.path_img, 'test')
    if len(test_only_1mm) > 0 or len(test_only_5mm) > 0:
        print(f'Warning: test set has unmatched cases, skipped. 1mm_only={len(test_only_1mm)}, 5mm_only={len(test_only_5mm)}')
    if len(test_list) == 0:
        raise RuntimeError(f'No paired test cases found under path_img={opt.path_img}')
    test_set = test_Dataset(test_list)
    test_data_num = len(test_set.img_list)
    test_batch = Data.DataLoader(dataset=test_set, batch_size=opt.val_bs, shuffle=False,
                                num_workers=opt.test_num_workers, worker_init_fn=worker_init_fn)
    print('load test data done, num =', test_data_num)

    load_net = save_dict['net']
    load_model_dict = load_net.state_dict()

    net = model_TransSR.TVSRN()
    net.load_state_dict(load_model_dict, strict=False)

    del save_dict
    net = net.to(device)
    net = net.eval()

    with torch.no_grad():
        pid_list = []
        psnr_list = []
        ssim_list = [] if compute_ssim else None

        for i, return_list in tqdm(enumerate(test_batch)):
            case_name, x, y, pos_list = return_list
            case_name = case_name[0]

            pid_list.append(case_name)

            x = x.squeeze().data.numpy()
            y = y.squeeze().data.numpy()

            y_pre = np.zeros_like(y)
            pos_list = pos_list.data.numpy()[0]

            for pos_idx, pos in enumerate(pos_list):
                tmp_x = x[pos_idx]
                tmp_pos_z, tmp_pos_y, tmp_pos_x = pos

                tmp_x = torch.from_numpy(tmp_x).unsqueeze(0).unsqueeze(0).float().to(device)
                tmp_y_pre = net(tmp_x, ratio=opt.ratio)
                tmp_y_pre = torch.clamp(tmp_y_pre, 0, 1)
                y_for_psnr = tmp_y_pre.data.squeeze().cpu().numpy()

                D = y_for_psnr.shape[0]
                crop_margin = getattr(opt, 'crop_margin', 3)
                pos_z_s = opt.ratio * tmp_pos_z + crop_margin
                pos_y_s = tmp_pos_y
                pos_x_s = tmp_pos_x

                y_pre[pos_z_s: pos_z_s+D, pos_y_s:pos_y_s +
                      opt.vc_y, pos_x_s:pos_x_s+opt.vc_x] = y_for_psnr

            del tmp_y_pre, tmp_x

            y_pre = y_pre[opt.ratio:-opt.ratio]
            y = y[opt.ratio:-opt.ratio]

            save_name_pre = save_output_folder + '%s_pre.nii.gz' % case_name
            output_pre = sitk.GetImageFromArray(y_pre)
            sitk.WriteImage(output_pre, save_name_pre)

            psnr = non_model.cal_psnr(y_pre, y)
            psnr_list.append(psnr)

            if compute_ssim:
                case_ssim = non_model.cal_ssim_volume(
                    y,
                    y_pre,
                    device=device,
                    batch_size=ssim_batch_size,
                    stride=ssim_stride,
                )
                ssim_list.append(case_ssim)

        print(np.mean(psnr_list))
        if compute_ssim:
            print(np.mean(ssim_list))


if __name__ == '__main__':
    import fire

    fire.Fire()
