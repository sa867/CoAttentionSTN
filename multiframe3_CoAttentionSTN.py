import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init

affine_par = True

def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, int(out_channels/2), 3, padding=1),
        nn.GroupNorm(int(out_channels/8), int(out_channels/2)),
        nn.ReLU(inplace=True),
        nn.Conv3d(int(out_channels/2), out_channels, 3, padding=1),
        nn.GroupNorm(int(out_channels/4), out_channels),
        nn.ReLU(inplace=True)
    )

def single_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1),
        nn.GroupNorm(int(out_channels/4), out_channels),
        nn.ReLU(inplace=True)
    )


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.conv = nn.Conv3d(256, depth, 1)
        self.bn_x = nn.GroupNorm(int(depth/4), depth)
        self.conv2d_0 = nn.Conv3d(256, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.GroupNorm(int(depth/4), depth)
        self.conv2d_1 = nn.Conv3d(256, depth, kernel_size=3, stride=1, padding=padding_series[0],
                                  dilation=dilation_series[0])
        self.bn_1 = nn.GroupNorm(int(depth/4), depth)
        self.conv2d_2 = nn.Conv3d(256, depth, kernel_size=3, stride=1, padding=padding_series[1],
                                  dilation=dilation_series[1])
        self.bn_2 = nn.GroupNorm(int(depth/4), depth)
        self.conv2d_3 = nn.Conv3d(256, depth, kernel_size=3, stride=1, padding=padding_series[2],
                                  dilation=dilation_series[2])
        self.bn_3 = nn.GroupNorm(int(depth/4), depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv3d(depth * 5, 256, kernel_size=3, padding=1)  # 512 1x1Conv
        self.bn = nn.GroupNorm(int(256/4), 256)
        self.prelu = nn.PReLU()
        # for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv3d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)  # classes
        Bn = nn.GroupNorm(int(256/4), 256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)

    def forward(self, x):
        # out = self.conv2d_list[0](x)
        # mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size = x.shape[2:]
        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features = F.upsample(image_features, size=size, mode='trilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0)
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1)
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2)
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3)
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        # for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)

        return out

class encoder(nn.Module):
    def __init__(self, all_channel):  # 473./8=60
        super(encoder, self).__init__()
        self.dconv_down1 = double_conv(1, 64)
        self.dconv_down2 = double_conv(64, 128)
        self.dconv_down3 = double_conv(128, all_channel)
        self.maxpool = nn.MaxPool3d(2)

    def forward(self, input):
        input1_conv1 = self.dconv_down1(input)
        x1 = self.maxpool(input1_conv1)  # 64 -> 32
        input1_conv2 = self.dconv_down2(x1)
        x1 = self.maxpool(input1_conv2)  # 32 -> 16
        features = self.dconv_down3(x1)

        return features

class attention(nn.Module):
    def __init__(self, all_channel, all_dim=60 * 60):  # 473./8=60
        super(attention, self).__init__()
        self.linear_e = nn.Linear(all_channel, all_channel, bias=False)
        self.linear_q = nn.Linear(all_channel, all_channel, bias=False)
        self.channel = all_channel
        self.dim = all_dim
        self.gate = nn.Conv3d(256, 1, kernel_size=1, bias=False)
        self.gate_s = nn.Sigmoid()
        #self.gate_s = nn.Softmax()

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)


    def forward(self, exemplar, query):
        # Multi-frame spatiotemporal attention

        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1] * fea_size[2]
        exemplar_flat = exemplar.view(-1, query.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        query_t = torch.transpose(query_flat, 1, 2).contiguous()
        exemplar_corr = self.linear_e(exemplar_t)
        query_corr = self.linear_q(query_t)
        A1 = torch.bmm(exemplar_corr, query_flat)
        A2 = torch.bmm(query_corr, exemplar_flat)
        B1 = F.softmax(torch.transpose(A1, 1, 2), dim=1)    #correlation matrix
        B2 = F.softmax(torch.transpose(A2, 1, 2), dim=1)
        exemplar_att = torch.bmm(exemplar_flat, B1).contiguous()   #post-attention
        query_att = torch.bmm(query_flat, B2).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1], fea_size[2])
        input2_att1 = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1], fea_size[2])
        input1_mask = self.gate(input1_att)
        input2_mask = self.gate(input2_att1)
        input1_mask = self.gate_s(input1_mask)
        input2_mask = self.gate_s(input2_mask)

        #input1_att = input1_att * input1_mask
        #input2_att = input2_att1 * input2_mask

        return input1_mask, input2_mask  #input1_att, input1_att#, i


class UNet3DModel(nn.Module):
    def __init__(self):#, block, layers, num_classes, all_channel=256, all_dim=60 * 60):  # 473./8=60
        super(UNet3DModel, self).__init__()
        self.dconv_down1 = double_conv(2, 128)
        self.dconv_down2 = double_conv(128, 256)
        self.dconv_down3 = double_conv(256, 512)
        self.dconv_same = double_conv(512, 512)
        self.dconv_up3 = double_conv(512 + 512, 256)
        self.dconv_up2 = double_conv(256 + 256, 128)
        self.dconv_up1 = double_conv(128 + 128, 64)
        self.conv_last = nn.Conv3d(64, 3, kernel_size=3, padding=1)
        self.softmax = nn.Sigmoid()
        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)

    def forward(self, input):
        conv1 = self.dconv_down1(input)
        x = self.maxpool(conv1)   #64 -> 32

        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)   #32 -> 16

        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)   #16 -> 8

        x = self.dconv_same(x)

        x = self.upsample(x)      #8 -> 16
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)      #16 -> 32
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)

        x = self.upsample(x)      #32 -> 64
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)

        x = self.conv_last(x)

        return x


class FtoFAttentionModel(nn.Module):
    def __init__(self, all_channel=256, all_dim=60 * 60):  # 473./8=60
        super(FtoFAttentionModel, self).__init__()
        self.coattention_encoder = encoder(256)#ResNet(Bottleneck, [3, 4, 23, 3], 1)
        self.unet = UNet3DModel()
        self.softmax = nn.Sigmoid()
        self.prelu = nn.ReLU(inplace=True)
        self.att_comb = nn.Conv3d(2, 1, kernel_size=3, stride=1)

        self.attention = attention(256)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, input):

        input_size = input.size()[2:]
        exemplar = self.coattention_encoder(input[:,0,:,:,:].unsqueeze(1))
        intermediate1 = self.coattention_encoder(input[:,1,:,:,:].unsqueeze(1))

        query = self.coattention_encoder(input[:,2,:,:,:].unsqueeze(1))

        exemplar_att1, interm1_att = self.attention(exemplar, intermediate1)
        interm2_att, query_att1 = self.attention(intermediate1, query)
        exemplar_att2, query_att2 = self.attention(exemplar, query)

        exemplar_mask = torch.cat([exemplar_att1, exemplar_att2], dim=1)
        exemplar_mask = self.att_comb(exemplar_mask)
        #exemplar_mask = self.softmax(exemplar_mask)
        query_mask = torch.cat([query_att1, query_att2], dim=1)
        query_mask = self.att_comb(query_mask)
        #query_mask = self.softmax(query_mask)
        interm_mask = torch.cat([interm1_att, interm2_att], dim=1)
        interm_mask = self.att_comb(interm_mask)


        exemplar_mask = F.upsample(exemplar_mask, input_size, mode='trilinear')
        query_mask = F.upsample(query_mask, input_size, mode='trilinear')
        interm_mask = F.upsample(interm_mask, input_size, mode='trilinear')

        frame1 = input[:, 0, :, :, :].unsqueeze(1) * exemplar_mask
        frame_mid = input[:, 1, :, :, :].unsqueeze(1) * interm_mask
        frame2 = input[:, 2, :, :, :].unsqueeze(1) * query_mask

        pred_disp_ES_ED = self.unet(torch.cat([frame1, frame2], dim=1))
        pred_disp12 = self.unet(torch.cat([frame1, frame_mid], dim=1))
        pred_disp23 = self.unet(torch.cat([frame_mid, frame2], dim=1))

        return pred_disp_ES_ED, frame1, frame2, frame_mid, exemplar_mask, query_mask, interm_mask, pred_disp12, pred_disp23