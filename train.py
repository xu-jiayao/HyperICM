"""Per-image semantic-guided encoder adaptation for HyperICM."""

import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
import random
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
import argparse
import gc

if __package__:
    from .evaluating import run_eval
    from .Hyperfree import SamAutomaticMaskGenerator, HyperFree_model_registry
    from .hypericm.checkpoints import checkpoint_state, load_codec_checkpoint
    from .hypericm.datasets import DATASET_NAMES, get_dataset_spec
    from .hypericm.patches import HyperspectralPatchDataset
    from .model import DMC, DMCI
    from .Hyperfree_test import HyperspectralTester
else:  # ``python train.py`` from the release directory
    from evaluating import run_eval
    from Hyperfree import SamAutomaticMaskGenerator, HyperFree_model_registry
    from hypericm.checkpoints import checkpoint_state, load_codec_checkpoint
    from hypericm.datasets import DATASET_NAMES, get_dataset_spec
    from hypericm.patches import HyperspectralPatchDataset
    from model import DMC, DMCI
    from Hyperfree_test import HyperspectralTester


ROOT = Path(__file__).resolve().parent
DEVICE = torch.device("cpu")
DEVICE_HYPERFREE = torch.device("cpu")
PARALLEL_DEVICES = []
HyperFree = None
mask_generator = None

##------------------hyperfree related------------------
wavelengths_hyper=[ 404.6129, 414.2946, 423.9808, 433.6713, 443.3662, 453.0655, 
        462.7692, 472.4773, 482.1898, 491.9066, 501.6279, 511.3535, 521.0836, 530.818, 540.5568, 550.3, 
        560.0477, 569.7996, 579.556, 589.3168, 599.0819, 608.8515, 618.6254, 628.4037, 638.1865, 647.9736, 
        657.7651, 667.561, 655.2923, 665.0994, 674.9012, 684.6979, 694.4894, 704.2756, 714.0566, 723.8325, 
        733.6031, 743.3685, 753.1287, 762.8837, 772.6335, 782.3781, 792.1174, 801.8516, 811.5805, 821.3043, 
        831.0228, 840.7361, 850.4442, 860.1471, 869.8448, 879.5372, 889.2245, 898.9066, 908.5834, 918.2551, 
        927.9214, 937.5827, 947.2387, 956.8895, 966.5351, 976.1755, 985.8106, 995.4406, 1005.065, 1014.685, 
        1024.299, 1033.908, 1043.512, 1053.111, 1062.704, 1072.293, 1081.876, 1091.454, 1101.026, 1110.594, 
        1120.156, 1129.713, 1139.265, 1148.811, 1158.353, 1167.889, 1177.42, 1186.946, 1196.466, 1205.982, 
        1215.492, 1224.997, 1234.496, 1243.991, 1253.48, 1262.964, 1253.373, 1263.346, 1273.318, 1283.291, 
        1293.262, 1303.234, 1313.206, 1323.177, 1333.148, 1343.119, 1353.089, 1363.06, 1373.03, 1383.0, 
        1392.969, 1402.939, 1412.908, 1422.877, 1432.845, 1442.814, 1452.782, 1462.75, 1472.718, 1482.685, 
        1492.652, 1502.619, 1512.586, 1522.552, 1532.518, 1542.484, 1552.45, 1562.416, 1572.381, 1582.346, 
        1592.311, 1602.275, 1612.24, 1622.204, 1632.167, 1642.131, 1652.094, 1662.057, 1672.02, 1681.983, 
        1691.945, 1701.907, 1711.869, 1721.831, 1731.792, 1741.753, 1751.714, 1761.675, 1771.635, 1781.596, 
        1791.556, 1801.515, 1811.475, 1821.434, 1831.393, 1841.352, 1851.31, 1861.269, 1871.227, 1880.184, 
        1874.164, 1884.225, 1894.285, 1904.342, 1914.396, 1924.448, 1934.499, 1944.546, 1954.592, 1964.635, 
        1974.675, 1984.714, 1994.75, 2004.784, 2014.815, 2024.845, 2034.872, 2044.896, 2054.919, 2064.939, 
        2074.956, 2084.972, 2094.985, 2104.996, 2115.004, 2125.01, 2135.014, 2145.016, 2155.015, 2165.012, 
        2175.007, 2184.999, 2194.989, 2204.977, 2214.962, 2224.945, 2234.926, 2244.905, 2254.881, 2264.854, 
        2274.826, 2284.795, 2294.762, 2304.727, 2314.689, 2324.649, 2334.607, 2344.562, 2354.516, 2364.467, 
        2374.415, 2384.361, 2394.305, 2404.247, 2414.186, 2424.123, 2434.058, 2443.99, 2453.92, 2463.848, 
        2473.773, 2483.696, 2493.617, 2503.536]

wavelengths1 = [
    401.809998, 404.031006, 406.252014, 408.472992, 410.694000, 412.915009,
    415.135986, 417.356995, 419.578003, 421.799988, 424.020996, 426.242004,
    428.463013, 430.683990, 432.904999, 435.126007, 437.346985, 439.567993,
    441.789001, 444.010010, 446.230988, 448.451996, 450.674011, 452.894989,
    455.115997, 457.337006, 459.558014, 461.778992, 464.000000, 466.221008,
    468.441986, 470.662994, 472.884003, 475.105011, 477.326996, 479.548004,
    481.769012, 483.989990, 486.210999, 488.432007, 490.653015, 492.873993,
    495.095001, 497.316010, 499.536987, 501.757996, 503.979004, 506.200989,
    508.421997, 510.643005, 512.864014, 515.085022, 517.306030, 519.526978,
    521.747986, 523.968994, 526.190002, 528.411011, 530.632019, 532.854004,
    535.075012, 537.296021, 539.517029, 541.737976, 543.958984, 546.179993,
    548.401001, 550.622009, 552.843018, 555.064026, 557.284973, 559.505981,
    561.728027, 563.948975, 566.169983, 568.390991, 570.612000, 572.833008,
    575.054016, 577.275024, 579.495972, 581.716980, 583.937988, 586.158997,
    588.380981, 590.601990, 592.822998, 595.044006, 597.265015, 599.486023,
    601.706970, 603.927979, 606.148987, 608.369995, 610.591003, 612.812012,
    615.033020, 617.255005, 619.476013, 621.697021, 623.918030, 626.138977,
    628.359985, 630.580994, 632.802002, 635.023010, 637.244019, 639.465027,
    641.685974, 643.908020, 646.129028, 648.349976, 650.570984, 652.791992,
    655.013000, 657.234009, 659.455017, 661.676025, 663.896973, 666.117981,
    668.338989, 670.559998, 672.781982, 675.002991, 677.223999, 679.445007,
    681.666016, 683.887024, 686.107971, 688.328979, 690.549988, 692.770996,
    694.992004, 697.213013, 699.434998, 701.656006, 703.877014, 706.098022,
    708.318970, 710.539978, 712.760986, 714.981995, 717.203003, 719.424011,
    721.645020, 723.866028, 726.086975, 728.309021, 730.530029, 732.750977,
    734.971985, 737.192993, 739.414001, 741.635010, 743.856018, 746.077026,
    748.297974, 750.518982, 752.739990, 754.961975, 757.182983, 759.403992,
    761.625000, 763.846008, 766.067017, 768.288025, 770.508972, 772.729980,
    774.950989, 777.171997, 779.393005, 781.614014, 783.835999, 786.057007,
    788.278015, 790.499023, 792.719971, 794.940979, 797.161987, 799.382996,
    801.604004, 803.825012, 806.046021, 808.267029, 810.489014, 812.710022,
    814.931030, 817.151978, 819.372986, 821.593994, 823.815002, 826.036011,
    828.257019, 830.478027, 832.698975, 834.919983, 837.140991, 839.362976,
    841.583984, 843.804993, 846.026001, 848.247009, 850.468018, 852.689026,
    854.909973, 857.130981, 859.351990, 861.572998, 863.794006, 866.015991,
    868.237000, 870.458008, 872.679016, 874.900024, 877.120972, 879.341980,
    881.562988, 883.783997, 886.005005, 888.226013, 890.447021, 892.668030,
    894.890015, 897.111023, 899.331970, 901.552979, 903.773987, 905.994995,
    908.216003, 910.437012, 912.658020, 914.879028, 917.099976, 919.320984,
    921.543030, 923.763977, 925.984985, 928.205994, 930.427002, 932.648010,
    934.869019, 937.090027, 939.310974, 941.531982, 943.752991, 945.973999,
    948.195007, 950.416992, 952.638000, 954.859009, 957.080017, 959.301025,
    961.521973, 963.742981, 965.963989, 968.184998, 970.406006, 972.627014,
    974.848022, 977.070007, 979.291016, 981.512024, 983.732971, 985.953979,
    988.174988, 990.395996, 992.617004, 994.838013, 997.059021, 999.280029]

wavelengths2 = [
  398.565002,  400.776001,  402.985992,  405.196014,  407.406006,  409.615997,
  411.825989,  414.036011,  416.246002,  418.455994,  420.665985,  422.876007,
  425.087006,  427.296997,  429.506989,  431.717010,  433.927002,  436.136993,
  438.346985,  440.557007,  442.766998,  444.976990,  447.187012,  449.398010,
  451.608002,  453.817993,  456.028015,  458.238007,  460.447998,  462.657990,
  464.868011,  467.078003,  469.287994,  471.497986,  473.708008,  475.919006,
  478.128998,  480.338989,  482.549011,  484.759003,  486.968994,  489.178986,
  491.389008,  493.598999,  495.808990,  498.019012,  500.230011,  502.440002,
  504.649994,  506.859985,  509.070007,  511.279999,  513.489990,  515.700012,
  517.909973,  520.119995,  522.330017,  524.541016,  526.750977,  528.960999,
  531.171021,  533.380981,  535.591003,  537.801025,  540.010986,  542.221008,
  544.431030,  546.640991,  548.851990,  551.062012,  553.271973,  555.481995,
  557.692017,  559.901978,  562.112000,  564.322021,  566.531982,  568.742004,
  570.952026,  573.163025,  575.372986,  577.583008,  579.793030,  582.002991,
  584.213013,  586.422974,  588.632996,  590.843018,  593.052979,  595.263000,
  597.473999,  599.684021,  601.893982,  604.104004,  606.314026,  608.523987,
  610.734009,  612.943970,  615.153992,  617.364014,  619.573975,  621.784973,
  623.994995,  626.205017,  628.414978,  630.625000,  632.835022,  635.044983,
  637.255005,  639.465027,  641.674988,  643.885010,  646.096008,  648.306030,
  650.515991,  652.726013,  654.935974,  657.145996,  659.356018,  661.565979,
  663.776001,  665.986023,  668.195984,  670.406982,  672.617004,  674.827026,
  677.036987,  679.247009,  681.456970,  683.666992,  685.877014,  688.086975,
  690.296997,  692.507019,  694.718018,  696.927979,  699.138000,  701.348022,
  703.557983,  705.768005,  707.978027,  710.187988,  712.398010,  714.607971,
  716.817993,  719.028015,  721.239014,  723.448975,  725.658997,  727.869019,
  730.078979,  732.289001,  734.499023,  736.708984,  738.919006,  741.129028,
  743.338989,  745.549988,  747.760010,  749.969971,  752.179993,  754.390015,
  756.599976,  758.809998,  761.020020,  763.229980,  765.440002,  767.650024,
  769.861023,  772.070984,  774.281006,  776.491028,  778.700989,  780.911011,
  783.120972,  785.330994,  787.541016,  789.750977,  791.960999,  794.171997,
  796.382019,  798.591980,  800.802002,  803.012024,  805.221985,  807.432007,
  809.642029,  811.851990,  814.062012,  816.271973,  818.482971,  820.692993,
  822.903015,  825.112976,  827.322998,  829.533020,  831.742981,  833.953003,
  836.163025,  838.372986,  840.583008,  842.794006,  845.004028,  847.213989,
  849.424011,  851.633972,  853.843994,  856.054016,  858.263977,  860.473999,
  862.684021,  864.893982,  867.104980,  869.315002,  871.525024,  873.734985,
  875.945007,  878.155029,  880.364990,  882.575012,  884.784973,  886.994995,
  889.205017,  891.416016,  893.625977,  895.835999,  898.046021,  900.255981,
  902.466003,  904.676025,  906.885986,  909.096008,  911.306030,  913.515991,
  915.726990,  917.937012,  920.146973,  922.356995,  924.567017,  926.776978,
  928.987000,  931.197021,  933.406982,  935.617004,  937.827026,  940.038025,
  942.247986,  944.458008,  946.668030,  948.877991,  951.088013,  953.297974,
  955.507996,  957.718018,  959.927979,  962.138000,  964.348022,  966.559021,
  968.768982,  970.979004,  973.189026,  975.398987,  977.609009,  979.818970,
  982.028992,  984.239014,  986.448975,  988.658997,  990.869995,  993.080017,
  995.289978,  997.500000,  999.710022, 1001.919983]

base_GSDs = {
    "WHU-Hi-LongKou": 0.463,
    "WHU-Hi-HanChuan": 0.109,
    "WHU-Hi-HongHu": 0.043
}

FEATURE_INDEX = [0,1,2,3,4]


def initialize_hyperfree(checkpoint_path, pred_iou_thresh=0.3, stability_score_thresh=0.4):
    global HyperFree, mask_generator
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing Hyperfree checkpoint: {checkpoint_path}")
    HyperFree = HyperFree_model_registry["vit_b"](
        checkpoint=str(checkpoint_path),
        encoder_global_attn_indexes=-1,
        merge_indexs=None,
    ).to(DEVICE_HYPERFREE)
    HyperFree.eval()
    for parameter in HyperFree.parameters():
        parameter.requires_grad_(False)
    mask_generator = SamAutomaticMaskGenerator(
        model=HyperFree,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        points_per_side=32,
    )

# @torch.no_grad()
def Hyperfree_inference(img, GSD, wavelengths):
    if mask_generator is None:
        raise RuntimeError("Hyperfree must be initialized before extracting features.")
    x_hat_ = img.to(DEVICE_HYPERFREE)
    gsd_value = torch.as_tensor(GSD, dtype=torch.float32, device=DEVICE_HYPERFREE).reshape(-1)[0]
    mask_generator.predictor.set_image(x_hat_.permute(1, 2, 0), True, wavelengths, 
                                    GSD=gsd_value.reshape(1))
    feat_out_all = mask_generator.predictor.model.image_encoder.multi_stage_features
    feat_out = [feat_out_all[i] for i in FEATURE_INDEX] # 
    return feat_out

def get_distortion(infeat, outfeat, mse):
    distortion = mse(infeat, outfeat)
    return distortion

def get_bpp(x_recon, likelihoods):  # Returns calculated bpp for train and test

    size_est = (-np.log(2) * x_recon.numel())
    bpp = torch.sum(torch.log(likelihoods)) / size_est
    return bpp

""" Revision Check """
q_index_to_lambda = {i: 1 for i in range(1, 65)} #{i: 1 + 24 * (i - 1) for i in range(1, 65)}
def train_one_epoch(args, i_frame_model, p_frame_model, dataset, optimizer, bpp_beta, feature_beta, mse_beta, writer, epoch, q):

    print("Training one epoch...")
    mse = torch.nn.MSELoss(reduction='mean')
    # weights= [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
    weights= [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
    num_count = 0
    total_loss_epoch = 0.0
    total_feature_loss_epoch = 0.0
    total_mse_epoch = 0.0
    total_bpp_epoch = 0.0
    feature_loss = 0.0
    #mse
    num_count = 0
    ''' All batches update '''
    optimizer.zero_grad()
    pbar = tqdm(dataset, desc="Training", dynamic_ncols=True, ncols=100)
    for img_patches, feature_tensors, GSD, max_val in pbar:
        
        img_patches = img_patches.to(DEVICE)

        batch_q = random.randint(1, 63) if q == 0 else q

        num_count += 1

        i_frame_model.train()
        p_frame_model.train()

        x_hats_all = []
        # x_test = []

        B, C, H, W = img_patches.shape   # B=1, C=224
        GROUP_SIZE = args.gop
        num_groups = C // GROUP_SIZE
        left_8 = C - num_groups*GROUP_SIZE
        
        img_patches_front = img_patches[:, :num_groups*GROUP_SIZE, :, :]
        grouped = img_patches_front.view(B*num_groups, GROUP_SIZE, H, W)
        
        bpp_loss_group = 0.0
        distortion_loss_group = 0.0

        dataset_spec = get_dataset_spec(args.dataset)
        wavelengths = dataset_spec.wavelengths
        base_GSD = dataset_spec.base_gsd
        ratio = 1024 / (max(H, W))
        GSD = base_GSD / ratio
        GSD = torch.tensor([GSD])

        for i in range(GROUP_SIZE):
            frame = grouped[:, i:i + 1, :, :]

            if i == 0:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = \
                        i_frame_model.forward(frame, batch_q)

            elif i == 1:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                        p_frame_model.forward(frame, x_hat.to(DEVICE), None, batch_q)

            else:
                x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                    p_frame_model.forward(frame, None, feature.to(DEVICE), batch_q)

            bpp_y = get_bpp(y_hat, y_likelihoods)
            bpp_z = get_bpp(z_hat, z_likelihoods)    

            distortion = get_distortion(frame, x_hat, mse)

            # ✅ 保存当前帧的结果
            x_hats_all.append(x_hat)
            # x_test.append(frame)

            bpp_loss_group += (bpp_y + bpp_z)/GROUP_SIZE
            distortion_loss_group += (distortion * weights[i] * q_index_to_lambda[batch_q])/GROUP_SIZE

        if left_8!=0:
            x_hats_tensor_front = torch.cat(x_hats_all, dim=0)
            x_hats_tensor_front = x_hats_tensor_front.view(B, num_groups * GROUP_SIZE, H, W)
            x_hats_all = [] 
            img_patches_back = img_patches[:, num_groups*GROUP_SIZE:, :, :]
            grouped = img_patches_back.view(B, left_8, H, W)
            for i in range(left_8):
                frame = grouped[:, i:i + 1, :, :]

                if i == 0:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = \
                            i_frame_model.forward(frame, batch_q)

                elif i == 1:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                            p_frame_model.forward(frame, x_hat.to(DEVICE), None, batch_q)

                else:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                        p_frame_model.forward(frame, None, feature.to(DEVICE), batch_q)

                bpp_y = get_bpp(y_hat, y_likelihoods)
                bpp_z = get_bpp(z_hat, z_likelihoods)    

                distortion = get_distortion(frame, x_hat, mse)

                # ✅ 保存当前帧的结果
                x_hats_all.append(x_hat)
                # x_test.append(frame)

                bpp_loss_group += (bpp_y + bpp_z)/left_8
                distortion_loss_group += (distortion * weights[i] * q_index_to_lambda[batch_q])/left_8

            # bpp_loss_group /= GROUP_SIZE;
            # distortion_loss_group /= GROUP_SIZE;
            
            # del y_hat, z_hat, y_likelihoods, z_likelihoods, distortion
            
            # =============================================
            # ============= feature loss 计算 ==============
            # =============================================
            # x_hats_tensor = torch.cat(x_hats_all, dim=1) 
            x_hats_tensor_end = torch.cat(x_hats_all, dim=0) 
            x_hats_tensor_end = x_hats_tensor_end.view(B, left_8, H, W)
            # x_hats_tensor = x_hats_tensor_front.view(B, num_groups * GROUP_SIZE, H, W)
            x_hats_tensor = torch.cat([x_hats_tensor_front, x_hats_tensor_end], dim=1)
            del x_hats_tensor_front, x_hats_tensor_end, x_hats_all
        else:
            x_hats_tensor = torch.cat(x_hats_all, dim=0) 
            x_hats_tensor = x_hats_tensor.view(B, num_groups * GROUP_SIZE, H, W)
            del x_hats_all

        del img_patches, grouped

        feat_out_b = []
        for i_num in range(B):
            feats=Hyperfree_inference(x_hats_tensor[i_num], GSD, wavelengths)
            feats = [f.unsqueeze(0) if f.dim() == 4 else f for f in feats]  # 保证有 batch 维
            feat_out_b.append(feats)
        feat_out = [
            torch.cat([feat_out_b[b][i] for b in range(B)], dim=0)
            for i in range(len(feat_out_b[0]))
        ]
  
        feature_tensors = [f.to(DEVICE_HYPERFREE) for f in feature_tensors]
        
        del feat_out_b

        feature_loss = torch.stack([mse(f_out, f_tar) for f_out, f_tar in zip(feat_out, feature_tensors)]).mean().to(DEVICE)

        del feature_tensors, feat_out

        # =============================================total loss calculation
        final_loss_feature = (bpp_loss_group * bpp_beta + feature_loss * feature_beta + distortion_loss_group * mse_beta) * thresh #feature_loss * feature_beta * thresh #
        

        current_lr = optimizer.param_groups[0]["lr"]
        ''' change '''
        final_loss = final_loss_feature / len(dataset)
        (final_loss).backward()
        # optimizer.zero_grad()
        # (final_loss_feature).backward()
        # optimizer.step()

        # ===== 累计 =====
        total_loss_epoch += final_loss_feature.item()

        total_bpp_epoch += bpp_loss_group.item()
        
        total_mse_epoch += distortion_loss_group.item()  #mse
        
        total_feature_loss_epoch += feature_loss.item()
        #mse

        pbar.set_postfix({
                "loss": f"{total_loss_epoch / num_count:.4f}",
                
                "bpp": f"{total_bpp_epoch / num_count:.4f}",
                
                "mse": f"{total_mse_epoch / num_count:.6f}",
                
                "feature_loss": f"{total_feature_loss_epoch / num_count:.4f}",
                
                "lr": f"{current_lr:.6f}",
                "qp": f"{batch_q}"
            }) #   
    ''' change '''
    optimizer.step()
    optimizer.zero_grad()
    print(f"✅ Epoch Done."
          f"avg_total_loss={total_loss_epoch / num_count:.4f}, "

          f"avg_bpp={total_bpp_epoch / num_count:.4f}, "

          f"avg_feature={total_feature_loss_epoch / num_count:.4f}," 

          f"avg_mse={total_mse_epoch / num_count:.6f},") #  
    
    writer.add_scalar('train_total_loss', total_loss_epoch / num_count, epoch)

    writer.add_scalar('train_loss_bpp', total_bpp_epoch / num_count, epoch)
    writer.add_scalar('train_loss_feature', total_feature_loss_epoch / num_count, epoch)

    writer.add_scalar('train_loss_mse', total_mse_epoch / num_count, epoch)


def test_one_epoch(args, i_frame_model, p_frame_model, dataset, epoch, q):
    print("Testing one epoch...")

    mse = torch.nn.MSELoss(reduction='mean')
    GROUP_SIZE = args.gop
    NUM_BANDS = 224
    num_groups = NUM_BANDS // GROUP_SIZE

    total_bpp = 0.0

    total_mse = 0.0

    total_feat_loss = 0.0
    total_q = 0.0
    num_count = 0

    i_frame_model.eval()
    p_frame_model.eval()

    pbar = tqdm(dataset, desc=f"Testing (epoch {epoch})", dynamic_ncols=True, ncols=100)
    # weights= [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
    weights= [0.5, 1.2, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9, 0.5, 0.9]
    with torch.no_grad():
        for img_patches, feature_tensors, GSD, max_val in pbar:
            
            num_count += 1
            batch_q = random.randint(1, 63) if q == 0 else q
            # q = 21
            dataset_spec = get_dataset_spec(args.dataset)
            wavelengths = dataset_spec.wavelengths

            img_patches = img_patches.to(DEVICE)
            """
            B, C, H, W = img_patches.shape   # B=1, C=224
            num_groups = C // GROUP_SIZE
            grouped = img_patches.view(B*num_groups, GROUP_SIZE, H, W) #

            x_hats_all = []

            batch_bpp = 0.0

            batch_mse = 0.0

            # ===== 遍历每个子波段 =====
            for i in range(GROUP_SIZE):
                frame = grouped[:, i:i + 1, :, :]

                if i == 0:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = \
                            i_frame_model.forward(frame, batch_q)
                  
                elif i == 1:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                            p_frame_model.forward(frame, x_hat, None, q)
                    
                else:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                        p_frame_model.forward(frame, None, feature, q)
                    
                # ===== BPP =====
                bpp_y = get_bpp(y_hat, y_likelihoods)
                bpp_z = get_bpp(z_hat, z_likelihoods)
                batch_bpp += bpp_y + bpp_z

                # ===== distortion =====
                distortion_loss = get_distortion(frame, x_hat, mse)
                batch_mse += distortion_loss

                # 保存
                x_hats_all.append(x_hat)

            # ===== 平均 =====
            batch_bpp /= GROUP_SIZE
            batch_mse /= (num_groups * GROUP_SIZE)

            # ===== Feature Loss visual =====
            x_hats_tensor = torch.cat(x_hats_all, dim=1) 
            x_hats_tensor = x_hats_tensor.view(B, num_groups * GROUP_SIZE, H, W)"""
            
            B, C, H, W = img_patches.shape   # B=1, C=224
            # GROUP_SIZE = 8
            num_groups = C // GROUP_SIZE
            left_8 = C - num_groups*GROUP_SIZE

            x_hats_all = []
            
            img_patches_front = img_patches[:, :num_groups*GROUP_SIZE, :, :]
            grouped = img_patches_front.view(B*num_groups, GROUP_SIZE, H, W)
            
            bpp_loss_group = 0.0
            distortion_loss_group = 0.0
            
            dataset_spec = get_dataset_spec(args.dataset)
            wavelengths = dataset_spec.wavelengths
            base_GSD = dataset_spec.base_gsd
            ratio = 1024 / (max(H, W))
            GSD = base_GSD / ratio
            GSD = torch.tensor([GSD]) 

            for i in range(GROUP_SIZE):
                frame = grouped[:, i:i + 1, :, :]

                if i == 0:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = \
                            i_frame_model.forward(frame, batch_q)

                elif i == 1:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                            p_frame_model.forward(frame, x_hat.to(DEVICE), None, batch_q)

                else:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                        p_frame_model.forward(frame, None, feature.to(DEVICE), batch_q)

                bpp_y = get_bpp(y_hat, y_likelihoods)
                bpp_z = get_bpp(z_hat, z_likelihoods)    

                distortion = get_distortion(frame, x_hat, mse)

                # ✅ 保存当前帧的结果
                x_hats_all.append(x_hat)

                bpp_loss_group += (bpp_y + bpp_z)/GROUP_SIZE
                distortion_loss_group += (distortion * weights[i] * q_index_to_lambda[batch_q])/GROUP_SIZE

            img_patches_back = img_patches[:, num_groups*GROUP_SIZE:, :, :]
            grouped = img_patches_back.view(B, left_8, H, W)
            for i in range(left_8):
                frame = grouped[:, i:i + 1, :, :]

                if i == 0:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods = \
                            i_frame_model.forward(frame, batch_q)

                elif i == 1:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                            p_frame_model.forward(frame, x_hat.to(DEVICE), None, batch_q)

                else:
                    x_hat, y_hat, y_likelihoods, z_hat, z_likelihoods, feature = \
                        p_frame_model.forward(frame, None, feature.to(DEVICE), batch_q)

                bpp_y = get_bpp(y_hat, y_likelihoods)
                bpp_z = get_bpp(z_hat, z_likelihoods)    

                distortion = get_distortion(frame, x_hat, mse)
                x_hats_all.append(x_hat)

                bpp_loss_group += (bpp_y + bpp_z)/left_8
                distortion_loss_group += (distortion * weights[i] * q_index_to_lambda[batch_q])/left_8

            # bpp_loss_group /= GROUP_SIZE;
            # distortion_loss_group /= GROUP_SIZE;
            batch_bpp = bpp_loss_group
            batch_mse = distortion_loss_group
            
            # del y_hat, z_hat, y_likelihoods, z_likelihoods, distortion
            
            # =============================================
            # ============= feature loss 计算 ==============
            # =============================================
            # x_hats_tensor = torch.cat(x_hats_all, dim=1) 
            x_hats_tensor = torch.cat(x_hats_all, dim=0) 
            x_hats_tensor = x_hats_tensor.view(B, num_groups * GROUP_SIZE+left_8, H, W)
            
            
            feat_out_b = []
            for i_num in range(B):
                feats=Hyperfree_inference(x_hats_tensor[i_num], GSD, wavelengths)
                feats = [f.unsqueeze(0) if f.dim() == 4 else f for f in feats]  # 保证有 batch 维
                feat_out_b.append(feats)
            feat_out = [
                torch.cat([feat_out_b[b][i] for b in range(B)], dim=0)
                for i in range(len(feat_out_b[0]))
            ]
            # feat_out = Hyperfree_inference(x_hats_tensor, GSD)
            # feat_out = [f.to(DEVICE_HYPERFREE) for f in feat_out]
            
            feature_tensors = [f.to(DEVICE_HYPERFREE) for f in feature_tensors]
            feature_loss = torch.stack([mse(f_out, f_tar) for f_out, f_tar in zip(feat_out, feature_tensors)]).mean().to(DEVICE)

            # ===== 累计 =====
            total_bpp += batch_bpp.item()
            total_feat_loss += feature_loss.item()

            total_mse += batch_mse.item()
            
            total_q += batch_q

            pbar.set_postfix({
                "bpp": f"{batch_bpp.item():.4f}",
                "feat_loss": f"{feature_loss.item():.6f}",
                "mse_visual": f"{batch_mse.item():.6f}",
                "qp": f"{batch_q}"
            })
            #    

    # ===== 平均指标 =====
    avg_bpp = total_bpp / num_count
    avg_feat_loss = total_feat_loss / num_count

    avg_mse = total_mse / num_count
    
    avg_q = total_q / num_count

    print(
        f"[Epoch {epoch}] Test avg bpp = {avg_bpp:.4f}, "
        f"avg feature loss = {avg_feat_loss:.6f}, "
        f"avg mse = {avg_mse:.6f}, "
        f"avg qp = {avg_q:.2f}"
    )

    return avg_bpp, avg_mse, avg_feat_loss, avg_q

def parse_args(argv):
    parser = argparse.ArgumentParser(description="HyperICM online encoder adaptation")
    parser.add_argument("-c", "--config", type=Path, help="Optional YAML configuration file")
    parser.add_argument("--exp-name", "--exp_name", dest="exp_name")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "online_adaptation")
    parser.add_argument("--patch-dir", type=Path)
    parser.add_argument("--base-checkpoint", "--checkpoint", dest="base_checkpoint", type=Path)
    parser.add_argument("--hyperfree-checkpoint", type=Path)
    parser.add_argument("--full-image", type=Path)
    parser.add_argument("--ground-truth", type=Path)

    parser.add_argument("--dataset", choices=DATASET_NAMES, default="WHU-Hi-LongKou")
    parser.add_argument("--lr", "--learning-rate", dest="lr", type=float, default=1e-4)
    parser.add_argument("--patch-size", "--patch_size", dest="patch_size", type=int, default=256)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=1)
    parser.add_argument(
        "--test-batch-size", "--test_Batch_size", dest="test_batch_size", type=int, default=1
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--qp", type=int, default=21)
    parser.add_argument("--eval-qp", type=int, help="Concrete QP for full-image evaluation (defaults to --qp or 21).")
    parser.add_argument("--gop", type=int, default=8)
    parser.add_argument(
        "--loss-weights", "--para", dest="para", type=float, nargs=4,
        default=[0.5, 0.06, 5.0, 10.0],
        metavar=("SEMANTIC", "MSE", "RATE", "SCALE"),
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=28)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--date")

    default_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    parser.add_argument("--device", default=default_device)
    parser.add_argument("--hyperfree-device", default=default_device)
    parser.add_argument("--data-parallel-devices", type=int, nargs="+")
    parser.add_argument("--run-downstream", action="store_true")
    parser.add_argument("--feature-index", type=int, choices=range(5), default=1)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.2)
    parser.add_argument("--stability-score-thresh", type=float, default=0.3)
    parser.add_argument("--open-set-threshold", type=float, default=0.0)
    parser.add_argument("--resume-optimizer", action="store_true")

    config_args, _ = parser.parse_known_args(argv)
    if config_args.config is not None:
        config_path = config_args.config.resolve()
        if not config_path.is_file():
            parser.error(f"Configuration file not found: {config_path}")
        with config_path.open("r", encoding="utf-8") as handle:
            yaml_data = yaml.safe_load(handle) or {}
        if not isinstance(yaml_data, dict):
            parser.error("The YAML configuration root must be a mapping.")

        aliases = {"learning_rate": "lr", "loss_weights": "para"}
        yaml_data = {aliases.get(key, key): value for key, value in yaml_data.items()}
        for key in (
            "output_dir", "patch_dir", "base_checkpoint", "hyperfree_checkpoint",
            "full_image", "ground_truth",
        ):
            value = yaml_data.get(key)
            if value is not None:
                path = Path(value)
                yaml_data[key] = path if path.is_absolute() else (config_path.parent / path).resolve()
        parser.set_defaults(**yaml_data)
    args = parser.parse_args(argv)

    for option, value in (("--patch-dir", args.patch_dir), ("--base-checkpoint", args.base_checkpoint),
                          ("--hyperfree-checkpoint", args.hyperfree_checkpoint)):
        if value is None:
            parser.error(f"{option} is required (directly or through the YAML config).")
    if args.run_downstream and (args.full_image is None or args.ground_truth is None):
        parser.error("--run-downstream requires --full-image and --ground-truth.")
    if not 0 <= args.qp <= 63:
        parser.error("--qp must be in [0, 63]; QP 0 enables random training QPs.")
    if args.eval_qp is not None and not 1 <= args.eval_qp <= 63:
        parser.error("--eval-qp must be in [1, 63].")
    if not 1 <= args.gop <= 12:
        parser.error("--gop must be in [1, 12].")
    for option, value in (("--epochs", args.epochs), ("--batch-size", args.batch_size),
                          ("--test-batch-size", args.test_batch_size), ("--patch-size", args.patch_size),
                          ("--save-every", args.save_every), ("--eval-every", args.eval_every)):
        if value < 1:
            parser.error(f"{option} must be at least 1.")
    if args.patch_size % 16:
        parser.error("--patch-size must be divisible by 16 for the codec downsampling path.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    if args.num_workers != 0:
        parser.error("--num-workers must be 0 because patches run GPU Hyperfree extraction on demand.")
    for option, value in (("--pred-iou-thresh", args.pred_iou_thresh),
                          ("--stability-score-thresh", args.stability_score_thresh)):
        if not 0.0 <= value <= 1.0:
            parser.error(f"{option} must be in [0, 1].")
    if not -1.0 <= args.open_set_threshold <= 1.0:
        parser.error("--open-set-threshold must be in [-1, 1].")
    return args

if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    print(args)

    DEVICE = torch.device(args.device)
    DEVICE_HYPERFREE = torch.device(args.hyperfree_device)
    if DEVICE.type == "cuda" and DEVICE.index is None:
        DEVICE = torch.device("cuda:0")
    if DEVICE_HYPERFREE.type == "cuda" and DEVICE_HYPERFREE.index is None:
        DEVICE_HYPERFREE = torch.device("cuda:0")
    for name, device in (("codec", DEVICE), ("Hyperfree", DEVICE_HYPERFREE)):
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(f"CUDA was requested for {name}, but CUDA is unavailable.")
            index = 0 if device.index is None else device.index
            if index >= torch.cuda.device_count():
                raise ValueError(f"Unavailable {name} device: {device}")

    if args.data_parallel_devices:
        if DEVICE.type != "cuda":
            raise ValueError("--data-parallel-devices requires a CUDA codec device.")
        PARALLEL_DEVICES = list(dict.fromkeys(args.data_parallel_devices))
        primary_index = 0 if DEVICE.index is None else DEVICE.index
        if PARALLEL_DEVICES[0] != primary_index:
            raise ValueError(
                "The first --data-parallel-devices value must match the index in --device."
            )
        unavailable = [index for index in PARALLEL_DEVICES if index >= torch.cuda.device_count()]
        if unavailable:
            raise ValueError(f"Unavailable data-parallel CUDA devices: {unavailable}")
    else:
        PARALLEL_DEVICES = []

    for label, path in (
        ("patch directory", args.patch_dir),
        ("base checkpoint", args.base_checkpoint),
        ("Hyperfree checkpoint", args.hyperfree_checkpoint),
    ):
        expected = path.is_dir() if label == "patch directory" else path.is_file()
        if not expected:
            raise FileNotFoundError(f"Missing {label}: {path}")
    if args.full_image is not None and not args.full_image.is_file():
        raise FileNotFoundError(f"Missing full evaluation image: {args.full_image}")
    if args.ground_truth is not None and not args.ground_truth.is_file():
        raise FileNotFoundError(f"Missing ground-truth raster: {args.ground_truth}")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    lr = args.lr
    qp = args.qp
    model_path = args.base_checkpoint
    feature_beta, mse_beta, bpp_beta, thresh = args.para
    print("feature beta is {}, mse beta is {}, bpp beta is {}, thresh is {}".format(feature_beta, mse_beta, bpp_beta, thresh))
    print("=================================")

    initialize_hyperfree(
        args.hyperfree_checkpoint,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    spec = get_dataset_spec(args.dataset)
    train_dataset = HyperspectralPatchDataset(
        args.patch_dir,
        spec,
        Hyperfree_inference,
        expected_size=args.patch_size,
    )
    test_dataset = train_dataset

    print("current feature indeces are {}".format(FEATURE_INDEX))

    timestamp = datetime.now().strftime("%Y%m%d")
    if args.exp_name:
        run_root = Path(f"{args.exp_name}-{timestamp}")
    else:
        run_root = args.output_dir / f"{args.dataset}-{timestamp}"
    save_path = run_root / (
        f"{args.para[0]}sam_{args.para[1]}mse_{args.para[2]}bpp_"
        f"{args.para[3]}thresh_{args.lr}lr"
    )
    save_path.mkdir(parents=True, exist_ok=True)
    print(f"save path is {save_path}")

    Init_Epoch = 0
    Fin_Epoch = args.epochs
    
    print("init epoch is {}, fin epoch is {}".format(Init_Epoch, Fin_Epoch))

    log_dir = save_path / "tensorboard"
    writer = SummaryWriter(str(log_dir))

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=DEVICE.type == "cuda",
    )

    print("test dataset length is {}".format(len(test_dataset)))
    print("train dataset length is {}".format(len(train_dataset)))
    print("=================================")

    i_frame_net = DMCI()
    p_frame_net = DMC()

    def set_trainable_para(net, trainable_keys=None):
        """
        Freeze all parameters except those in trainable_keys.
        trainable_keys: list of substrings
        """
        if trainable_keys is None:
            trainable_keys = []
        for k, p in net.named_parameters():
            p.requires_grad = False
            if any(k.startswith(key + '.') or k == key for key in trainable_keys):
                p.requires_grad = True
    set_trainable_para(i_frame_net, ['enc'])
    if args.gop != 1:
        set_trainable_para(p_frame_net, ['encoder'])
    """ Print Check """
    # print("Trainable parameters of i_frame_net:")
    # for k, p in i_frame_net.named_parameters():
    #     if p.requires_grad:
    #         print(k)
    # print("Trainable parameters of p_frame_net:")
    # for k, p in p_frame_net.named_parameters():
    #     if p.requires_grad:
    #         print(k)

    def get_combined_parameters(*models):
        parameters = []
        for model in models:
            parameters.extend([p for p in model.parameters() if p.requires_grad])
        return parameters
    combined_params = get_combined_parameters(i_frame_net, p_frame_net) #
    optimizer = torch.optim.Adam(combined_params, lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    def print_model_stats(models):
        total_params = 0
        trainable_params = 0
        for model in models:
            for p in model.parameters():
                num = p.numel()
                total_params += num
                if p.requires_grad:
                    trainable_params += num
        
        ratio = trainable_params / total_params if total_params > 0 else 0
        print("-" * 30)
        print(f"Total Parameters    : {total_params / 1e6:.4f} M") 
        print(f"Trainable Parameters: {trainable_params / 1e6:.4f} M") 
        print(f"Trainable Ratio     : {ratio * 100:.2f}%") 
        print("-" * 30)

    print_model_stats([i_frame_net, p_frame_net])

    print("=================================")
    print(f"Loading model from {model_path}")
    checkpoint = load_codec_checkpoint(model_path, i_frame_net, p_frame_net, strict=True)
    for metric_name in ("bpp", "mse", "feature_loss", "testdataset_featureloss"):
        if metric_name in checkpoint:
            print(f"{metric_name}: {checkpoint[metric_name]}")

    inet = i_frame_net.to(DEVICE)
    pnet = p_frame_net.to(DEVICE)
    if args.resume_optimizer:
        missing = {"optimizer", "lr_scheduler"}.difference(checkpoint)
        if missing:
            raise KeyError(f"Cannot resume optimizer; checkpoint is missing: {sorted(missing)}")
        optimizer.load_state_dict(checkpoint["optimizer"])
        lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
    if len(PARALLEL_DEVICES) > 1:
        inet = torch.nn.DataParallel(inet, device_ids=PARALLEL_DEVICES)
        pnet = torch.nn.DataParallel(pnet, device_ids=PARALLEL_DEVICES)
    print("=================================")

    evaluation_qp = args.eval_qp or (qp if qp != 0 else 21)

    def evaluate_checkpoint(checkpoint_path, step, prefix):
        if args.full_image is None:
            return
        run_name = f"{args.date or timestamp}-{prefix}-{step}"
        recon_path, bpsp, psnr = run_eval(
            date_str=run_name,
            model_path=checkpoint_path,
            gpus=DEVICE,
            test_img_path=args.full_image,
            metric_save_path="metrics.csv",
            gop=args.gop,
            qps=[evaluation_qp],
            save_root=save_path / "evaluation",
            models=(inet, pnet),
        )
        writer.add_scalar(f"{prefix}/bpsp", bpsp, step)
        writer.add_scalar(f"{prefix}/psnr", psnr, step)
        if not args.run_downstream:
            return
        downstream_dir = save_path / "downstream" / f"{prefix}-{step}"
        tester = HyperspectralTester({
            "dataset": args.dataset,
            "input": args.full_image,
            "gt_path": args.ground_truth,
            "recon_path": recon_path,
            "save_dir": downstream_dir,
            "ckpt_pth": args.hyperfree_checkpoint,
            "classifier": mask_generator,
            "method": "hyper",
            "csv_path": downstream_dir / "metrics.csv",
            "device": DEVICE_HYPERFREE,
            "qps": [evaluation_qp],
            "feature_index": args.feature_index,
            "open_set_threshold": args.open_set_threshold,
        })
        oa, aa, ka = tester.run_benchmark()
        writer.add_scalar(f"{prefix}/oa", oa, step)
        writer.add_scalar(f"{prefix}/aa", aa, step)
        writer.add_scalar(f"{prefix}/kappa", ka, step)
        del tester
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    evaluate_checkpoint(model_path, 0, "baseline")

    for epoch in range(Init_Epoch, Fin_Epoch):
        print("Epoch {}/{}".format(epoch, Fin_Epoch))

        train_one_epoch(args, inet, pnet, train_dataloader, optimizer, bpp_beta, feature_beta, mse_beta, writer, epoch, qp)

        print("Testing on test dataset:")
        bpp, avg_mse, avg_feat_loss, avg_q = test_one_epoch(args, inet, pnet, test_dataloader, epoch, qp)
        writer.add_scalar('testdataset_featureloss', avg_feat_loss, epoch)
        writer.add_scalar('testdataset_bpp', bpp, epoch)
        # writer.add_scalar('testdataset_mse', avg_mse, epoch)

        lr_scheduler.step()

        state = checkpoint_state(
            inet,
            pnet,
            epoch + 1,
            {
                "bpp": bpp,
                "mse": avg_mse,
                "feature_loss": avg_feat_loss,
                "testdataset_featureloss": avg_feat_loss,
                "average_qp": avg_q,
            },
            optimizer,
            lr_scheduler,
        )
        file_name = (
            f"gop{args.gop}-epoch_{epoch + 1}-bpp_{bpp:.6f}-distortion_{avg_mse:.6f}-"
            f"feat_loss_{avg_feat_loss:.6f}-avg_qp_{avg_q:.4f}.pth"
        )
        checkpoint_path = save_path / file_name
        should_evaluate = args.full_image is not None and (
            (epoch + 1) % args.eval_every == 0 or epoch == Fin_Epoch - 1
        )
        should_save = (
            (epoch + 1) % args.save_every == 0
            or should_evaluate
            or epoch == Fin_Epoch - 1
        )
        if should_save:
            torch.save(state, checkpoint_path)
        if should_evaluate:
            evaluate_checkpoint(checkpoint_path, epoch + 1, "online")

    writer.close()
