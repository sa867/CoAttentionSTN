import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from multiframe3_CoAttentionSTN import FtoFAttentionModel
import scipy.io as sio
from Porcine_Dataloader import PorcineDataset
import torch.nn.functional as F
import visdom
import math

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def CrossEntropyLoss(pred, label):
    #label = Variable(label.long()).cuda()
    criterion = torch.nn.BCELoss().cuda()

    return criterion(pred, label)

def generalized_diceLoss(pred, label):
    smooth = 1
    inputs = pred.view(-1)
    targets = label.view(-1)

    intersection = (inputs * targets).sum()
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

    return 1 - dice

def RegistrationLoss(img_fixed, img_moving, dx, dy, dz):
    #num_bats = img_moving.shape[0]  #number of samples in a batch
    num_rows = img_moving.shape[2]
    num_cols = img_moving.shape[3]
    num_depth = img_moving.shape[4]
    zv, yv, xv = torch.meshgrid([torch.arange(0, num_depth),
                                torch.arange(0, num_rows),
                                torch.arange(0, num_cols)])
    xv = xv.float().to(device)
    yv = yv.float().to(device)
    zv = zv.float().to(device)

    xv_moved = xv + dx[0, :, :, :]
    yv_moved = yv + dy[0, :, :, :]
    zv_moved = zv + dz[0, :, :, :]
    xv_moved_normalized = 2 * xv_moved / (num_cols - 1) - 1
    yv_moved_normalized = 2 * yv_moved / (num_rows - 1) - 1
    zv_moved_normalized = 2 * zv_moved / (num_depth - 1) - 1
    grid = torch.stack((xv_moved_normalized, yv_moved_normalized, zv_moved_normalized), 3)
    grid = grid.unsqueeze(0).float().to(device)
    out = F.grid_sample(img_moving.float(), grid, align_corners=True)

    return ((img_fixed - out)**2).mean()


class SpatialTransformer(nn.Module):
    def __init__(self, mode='bilinear'):
        super(SpatialTransformer, self).__init__()

        # Create the grid
        size = [64, 64, 64]
        self.mode = mode

        vectors = [torch.arange(0, s) for s in size]
        mesh_1, mesh_2, mesh_3 = torch.meshgrid(vectors)
        grid = torch.stack((mesh_1, mesh_2, mesh_3), 3)
        grid = grid.unsqueeze(0).float().to('cuda')

        self.register_buffer('grid', grid)
        self.sigmoid = nn.Sigmoid()


    def forward(self, src, flow):
        shape = flow.shape[2:]
        flow = flow.permute(0, 2, 3, 4, 1)
        new_loc = self.grid + flow

        # Need to normalize grid values to [-1, 1] for resampler
        new_loc[:, :, :, :, 0] = 2 * (new_loc[:, :, :, :, 0] / (shape[0] - 1) - 0.5)
        new_loc[:, :, :, :, 1] = 2 * (new_loc[:, :, :, :, 1] / (shape[1] - 1) - 0.5)
        new_loc[:, :, :, :, 2] = 2 * (new_loc[:, :, :, :, 2] / (shape[2] - 1) - 0.5)

        # Need to flip the channels
        new_loc = new_loc[..., [2, 1, 0]]

        return F.grid_sample(src, new_loc, align_corners=True, mode=self.mode)

class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return 1 - torch.mean(cc)

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Attention Network")

    # optimitization configuration
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to update the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=0.00001,
                        help="Base learning rate for training with polynomial decay.")  # 0.001
    parser.add_argument("--weight-decay", type=float, default=0.0005,
                        help="Regularization parameter for L2-loss.")  # 0.0005
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--power", type=float, default=0.9,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-seed", type=int, default=1234,
                        help="Random seed to have reproducible results.")

    return parser.parse_args()

args = get_arguments()


def configure_dataset_init_model(args):
    args.batch_size = 1  # 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
    args.maxEpoches = 100  # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),

def netParams(model):
    '''
    Computing total network parameters
    Args:
       model: model
    return: total network parameters
    '''
    total_parameters = 0
    for parameter in model.parameters():
        i = len(parameter.size())
        # print(parameter.size())
        p = 1
        for j in range(i):
            p *= parameter.size(j)
        total_parameters += p

    return total_parameters

def main():

    print("=====> Integrated Motion Tracking (Image Only) - Porcine")

    print("=====> Set GPU for training")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cudnn.enabled = True

    print("=====> Building network")
    model = FtoFAttentionModel()
    model.to(device)
    spatial_transform = SpatialTransformer()
    #ncc = NCC().loss
    mseloss = nn.MSELoss()

    cudnn.benchmark = True

    print('=====> Computing network parameters')
    total_parameters = netParams(model)
    print('Total network parameters: ' + str(total_parameters))
    print('Learning Rate: ' + str(args.learning_rate))
    maxEpoch = 100

    print("=====> Preparing training data")

    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
    optimizer.zero_grad()

    vis = visdom.Visdom()

    Y1 = torch.zeros((1)).cpu()
    X1 = torch.zeros((1)).cpu()

    loss_window = vis.line(
        Y=np.column_stack((Y1, Y1)),
        X=np.column_stack((X1, X1)),
        opts=dict(xlabel='epoch', ylabel='Loss', title='CoAttentionSTN+Temporal (3frame)',
                  legend=['training', 'validation']))

    print("=====> Preparing Dataloader")
    snapshot_dir = '/data/CoAttentionSTN_MedIA22/full_cycle_results_only_pstrain_model/new_trained_model_results'
    train_dataset = PorcineDataset('/data/Training_Data/MedIA2022_CoAttentionSTN/Porcine/full_cycle/train')
    val_dataset = PorcineDataset('/data/Training_Data/MedIA2022_CoAttentionSTN/Porcine/full_cycle/validation')
    params = {'batch_size': 1,
              'shuffle': True}
    training_generator = torch.utils.data.DataLoader(train_dataset, **params)
    validation_generator = torch.utils.data.DataLoader(val_dataset, **params)
    print('Total number of training images: ' + str(len(training_generator)))
    print('Total number of validation images: ' + str(len(validation_generator)))

    lowest_loss = 1e6
    print("=====> Begin to train")
    for epoch in range(0, maxEpoch):

        train_loss = 0.0
        val_loss = 0.0

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                np.random.seed(args.random_seed + epoch)

                for img, filename in validation_generator:
                    # Transfer to GPU
                    local_batch = img.to(device)#, myocardium1.to(device), myocardium2.to(device)

                    x_batch = local_batch.float()
                    #myo1 = myo1.float()
                    #myo2 = myo2.float()

                    x_batch /= 255

                    #myocardium1 = myo1.unsqueeze(1)
                    #myocardium2 = myo2.unsqueeze(1)

                    x_batch.requires_grad_()
                    image = Variable(x_batch)

                    optimizer.zero_grad()

                    pred_disp_es_ed, frame1, frame2, frame_mid, exemplar_mask, query_mask, interm_mask, pred_disp12, pred_disp23 = model(image)

                    deformed_img1 = spatial_transform(image[:, 0, :, :, :].unsqueeze(1), pred_disp_es_ed)
                    deformed_img2 = spatial_transform(frame1, pred_disp_es_ed)
                    deformed_img3 = spatial_transform(frame1, pred_disp12)
                    deformed_img4 = spatial_transform(frame_mid, pred_disp23)
                    deformed_int1 = spatial_transform(image[:, 0, :, :, :].unsqueeze(1), pred_disp12)
                    deformed_int2 = spatial_transform(image[:, 1, :, :, :].unsqueeze(1), pred_disp23)
                    deformed_attention_map1 = spatial_transform(exemplar_mask, pred_disp_es_ed)
                    deformed_attention_map2 = spatial_transform(exemplar_mask, pred_disp12)
                    deformed_attention_map3 = spatial_transform(interm_mask, pred_disp23)

                    loss_raw_img1 = mseloss(deformed_img1, image[:, 2, :, :, :].unsqueeze(1))
                    loss_raw_img2 = mseloss(deformed_int1, image[:, 1, :, :, :].unsqueeze(1))
                    loss_raw_img3 = mseloss(deformed_int2, image[:, 2, :, :, :].unsqueeze(1))

                    loss_att_img1 = mseloss(deformed_img2, frame2)
                    loss_att_img2 = mseloss(deformed_img3, frame_mid)
                    loss_att_img3 = mseloss(deformed_img4, frame2)

                    loss_attention1 = NCC().loss(deformed_attention_map1, query_mask)
                    loss_attention2 = NCC().loss(deformed_attention_map2, interm_mask)
                    loss_attention3 = NCC().loss(deformed_attention_map3, query_mask)

                    raw_img_loss = loss_raw_img1 + loss_raw_img2 + loss_raw_img3
                    att_img_loss = loss_att_img1 + loss_att_img2 + loss_att_img3
                    att_loss = loss_attention1 + loss_attention2 + loss_attention3
                    temp_cons = pred_disp12 + pred_disp23
                    temp_loss = mseloss(temp_cons, pred_disp_es_ed)

                    total_loss = raw_img_loss + att_img_loss + 0.02 * temp_loss + 0.05 * att_loss
                    total_loss.backward()
                    optimizer.step()
                    train_loss += total_loss.data

            else:
                model.eval()
                for img, filename in validation_generator:
                    # Transfer to GPU
                    local_batch = img.to(device)#, myocardium1.to(device), myocardium2.to(device)

                    x_batch = local_batch.float()
                    #myo1 = myo1.float()
                    #myo2 = myo2.float()

                    x_batch /= 255

                    #myocardium1 = myo1.unsqueeze(1)
                    #myocardium2 = myo2.unsqueeze(1)

                    x_batch.requires_grad_()
                    image = Variable(x_batch)

                    with torch.no_grad():
                        pred_disp_es_ed, frame1, frame2, frame_mid, exemplar_mask, query_mask, interm_mask, pred_disp12, pred_disp23 = model(image)

                    deformed_img1 = spatial_transform(image[:, 0, :, :, :].unsqueeze(1), pred_disp_es_ed)
                    deformed_img2 = spatial_transform(frame1, pred_disp_es_ed)
                    deformed_img3 = spatial_transform(frame1, pred_disp12)
                    deformed_img4 = spatial_transform(frame_mid, pred_disp23)
                    deformed_int1 = spatial_transform(image[:, 0, :, :, :].unsqueeze(1), pred_disp12)
                    deformed_int2 = spatial_transform(image[:, 1, :, :, :].unsqueeze(1), pred_disp23)
                    deformed_attention_map1 = spatial_transform(exemplar_mask, pred_disp_es_ed)
                    deformed_attention_map2 = spatial_transform(exemplar_mask, pred_disp12)
                    deformed_attention_map3 = spatial_transform(interm_mask, pred_disp23)

                    loss_raw_img1 = mseloss(deformed_img1, image[:, 2, :, :, :].unsqueeze(1))
                    loss_raw_img2 = mseloss(deformed_int1, image[:, 1, :, :, :].unsqueeze(1))
                    loss_raw_img3 = mseloss(deformed_int2, image[:, 2, :, :, :].unsqueeze(1))

                    loss_att_img1 = mseloss(deformed_img2, frame2)
                    loss_att_img2 = mseloss(deformed_img3, frame_mid)
                    loss_att_img3 = mseloss(deformed_img4, frame2)

                    loss_attention1 = NCC().loss(deformed_attention_map1, query_mask)
                    loss_attention2 = NCC().loss(deformed_attention_map2, interm_mask)
                    loss_attention3 = NCC().loss(deformed_attention_map3, query_mask)

                    raw_img_loss = loss_raw_img1 + loss_raw_img2 + loss_raw_img3
                    att_img_loss = loss_att_img1 + loss_att_img2 + loss_att_img3
                    att_loss = loss_attention1 + loss_attention2 + loss_attention3
                    temp_cons = pred_disp12 + pred_disp23
                    temp_loss = mseloss(temp_cons, pred_disp_es_ed)

                    total_loss = raw_img_loss + att_img_loss + 0.02 * temp_loss + 0.05 * att_loss
                    val_loss += total_loss.data

        scheduler.step()

        train_loss /= len(training_generator)
        val_loss /= len(validation_generator)
        vis.line(X=torch.ones((1, 1)).cpu() * epoch, Y=torch.Tensor([train_loss]).unsqueeze(0).cpu(), win=loss_window,
                 name='training',
                 update='append')

        vis.line(X=torch.ones((1, 1)).cpu() * epoch, Y=torch.Tensor([val_loss]).unsqueeze(0).cpu(), win=loss_window,
                 name='validation',
                 update='append')
        os.chdir(snapshot_dir)
        temp_epoch = epoch

        if phase == 'val' and val_loss < lowest_loss:
            lowest_loss = val_loss
            torch.save(model.state_dict(), osp.join(snapshot_dir, 'CoAttentionSTN_3frame_w_att_loss_full_cycle.pth'))
            file_name = 'CoAttentionSTN_3frame_w_att_loss_full_cycle.mat'
            image = image.cpu().detach().numpy()
            exemplar_mask = exemplar_mask.cpu().detach().numpy()
            query_mask = query_mask.cpu().detach().numpy()
            #int_mask = int_mask.cpu().detach().numpy()
            #myocardium1 = myocardium1.cpu().detach().numpy()
            #myocardium2 = myocardium2.cpu().detach().numpy()
            pred_disp_es_ed = pred_disp_es_ed.cpu().detach().numpy()
            #pred_disp12 = pred_disp12.cpu().detach().numpy()
            #pred_disp23 = pred_disp23.cpu().detach().numpy()
            deformed_img1 = deformed_img1.cpu().detach().numpy()
            deformed_img2 = deformed_img2.cpu().detach().numpy()
            frame1 = frame1.cpu().detach().numpy()
            frame2 = frame2.cpu().detach().numpy()

            sio.savemat(file_name, {'image': image, 'pred_disp_es_ed': pred_disp_es_ed, 'deformed1': deformed_img1,
                                     'deformed2': deformed_img2, 'frame1':frame1, 'frame2':frame2, 'epoch':temp_epoch,
                                    'exemplar_mask':exemplar_mask, 'query_mask':query_mask})
                                     #'myocardium1': myocardium1, 'myocardium2': myocardium2})
            print('Epoch: {0:3d} | train_loss: {1:2f} | val_loss: {2:2f} | Model Saved'.format(epoch, train_loss, val_loss))
        else:
            print('Epoch: {0:3d} | train_loss: {1:2f} | val_loss: {2:2f} |'.format(epoch, train_loss, val_loss))


if __name__ == '__main__':
    main()